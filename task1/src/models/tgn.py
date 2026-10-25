import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.models.tgn_support import BaseModel, load_weights, save_weights, current_run, init_run, TGNBPR, TGNCSE, TGNFCE
from tgn.model.tgn import TGN
from tgn.utils.data import Data, compute_time_statistics
from tgn.utils.utils import get_neighbor_finder

EDGE = False

class TGNModel(BaseModel):
    def __init__(
        self,
        name="tgn",
        memory_dim=64,
        n_degree=10,
        n_epochs=60,
        batch_size=1024,
        memory_batch_size=32,
        learning_rate=3e-4,
        device="cuda",
        seed=0,
        n_heads=2,
        dropout=0.1,
        n_layers=1,
        use_memory=True,
        message_dim=100,
        num_candidates=128,
        loss_type="bpr",
        item_chunk_size=128,
        backprop_every=32,
        patience=10,
        n_valid=5,
        edge=EDGE,
        trainable_node_features=True,
        # trainable_node_features=True,
        aggregator="attention",
        aggregation_half_life=600.0,
    ):
        super().__init__(name)
        self.name = name
        self.memory_dim = memory_dim
        self.n_degree = n_degree
        self.n_epochs = n_epochs
        self.batch_size = int(batch_size)
        self.memory_batch_size = int(memory_batch_size)
        self.gradient_accumulation_steps = (
            self.batch_size // self.memory_batch_size
        )
        self.backprop_every = int(backprop_every)
        self.learning_rate = 3e-4
        self.device = torch.device(device)
        self.seed = int(seed)
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.use_memory = use_memory
        self.message_dim = message_dim
        self.num_candidates = num_candidates
        self.loss_type = str(loss_type).lower()
        loss_classes = {
            "bpr": TGNBPR,
            "cse": TGNCSE,
            "fce": TGNFCE,
        }
        self.loss_function = loss_classes[self.loss_type]()
        self.item_chunk_size = int(item_chunk_size)
        self.patience = max(1, int(patience))
        self.n_valid = n_valid
        self.edge = EDGE if edge is None else bool(edge)
        self.trainable_node_features = bool(trainable_node_features)

        self.aggregator = aggregator
        # self.aggregation_half_life = float(aggregation_half_life)
        self.message_function = "identity"
        self.memory_update_at_start = True
        self.embedding_module = "graph_attention"
        self.memory_updater = "gru"

        self.model = None
        self.optimizer = None
        self.rng = np.random.RandomState(self.seed)
        self.n_users = self.n_items = 0

        self.best_epoch = None
        self.best_val_ndcg_20 = -np.inf
        self.best_state_dict = None
        self.best_train_memory = None
        self.test_metrics = None

    def _compute_full_catalog_logits(
        self, user_embeddings, memory, last_update, all_item_nodes, timestamps
    ):
        batch_size = len(timestamps)
        memory_input = (
            memory if memory is not None else user_embeddings.new_empty(0)
        )
        last_update_input = (
            last_update
            if last_update is not None
            else user_embeddings.new_empty(0)
        )
        catalog_logits = []

        for start in range(0, self.n_items, self.item_chunk_size):
            chunk_items = all_item_nodes[start : start + self.item_chunk_size]
            chunk_size = len(chunk_items)
            candidate_nodes = np.tile(chunk_items, batch_size)
            candidate_timestamps = np.repeat(timestamps, chunk_size)

            def score_chunk(
                users,
                memory_state,
                last_update_state,
                candidate_nodes=candidate_nodes,
                candidate_timestamps=candidate_timestamps,
                chunk_size=chunk_size,
            ):
                candidate_embeddings = self.model.compute_candidate_embeddings(
                    candidate_nodes,
                    candidate_timestamps,
                    memory_state if self.use_memory else None,
                    last_update_state if self.use_memory else None,
                    self.n_degree,
                ).view(batch_size, chunk_size, -1)
                return torch.einsum("bd,bid->bi", users, candidate_embeddings)

            catalog_logits.append(
                checkpoint(
                    score_chunk,
                    user_embeddings,
                    memory_input,
                    last_update_input,
                    use_reentrant=False,
                )
            )

        return torch.cat(catalog_logits, dim=1)

    def fit(
        self,
        train_dataset,
        val_dataset=None,
        metrics=None,
        max_top_n=20,
        checkpoint_path=None,
        resume_epoch=0,
    ):
        run = current_run()
        created_run = run is None

        if created_run:
            run = init_run(
                project=os.getenv("CLEARML_PROJECT", "recsys-btl"),
                name=os.getenv("CLEARML_TASK_NAME", f"tgn-{train_dataset.name}"),
                config={
                    "model": self.name,
                    "dataset": train_dataset.name,
                    "memory_dim": self.memory_dim,
                    "n_degree": self.n_degree,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "memory_batch_size": self.memory_batch_size,
                    "gradient_accumulation_steps": (
                        self.gradient_accumulation_steps
                    ),
                    "learning_rate": self.learning_rate,
                    "n_heads": self.n_heads,
                    "dropout": self.dropout,
                    "n_layers": self.n_layers,
                    "use_memory": self.use_memory,
                    "message_dim": self.message_dim,
                    "num_candidates": self.num_candidates,
                    "loss_type": self.loss_type,
                    "item_chunk_size": self.item_chunk_size,
                    "backprop_every": self.backprop_every,
                    "patience": self.patience,
                    "n_valid": self.n_valid,
                    "edge": self.edge,
                    "seed": self.seed,
                    "aggregator": self.aggregator,
                    # "aggregation_half_life": self.aggregation_half_life,
                },
            )
        run.define_metric("loss", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState(self.seed)
        self.best_epoch = None
        self.best_val_ndcg_20 = -np.inf
        self.best_state_dict = None
        self.best_train_memory = None

        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items
        candidate_items = np.asarray(getattr(train_dataset, "candidate_items", np.arange(self.n_items)), dtype=np.int64)

        train_df = (train_dataset._df.sort_values("timestamp", kind="mergesort").reset_index(drop=True))
        first_seen = np.full(self.n_items, np.inf)
        first_seen[train_df.groupby("item_id").timestamp.min().index.to_numpy(np.int64)] = train_df.groupby("item_id").timestamp.min().to_numpy()
        item_popularity = np.bincount(
            train_df["item_id"].to_numpy(),
            minlength=self.n_items
        )

        self.train_df = train_df.copy()
        source_nodes = train_df["user_id"].to_numpy(np.int64) + 1
        destination_nodes = (train_df["item_id"].to_numpy(np.int64) + self.n_users + 1)
        timestamps = train_df["timestamp"].to_numpy(np.float64)
        edge_idxs = np.arange(1, len(train_df) + 1, dtype=np.int64)
        if not self.edge:
            edge_idxs = np.zeros(len(train_df), dtype=np.int64)
        num_nodes = self.n_users + self.n_items + 1
        # node_features = self.rng.normal(
        #     size=(num_nodes, self.memory_dim)
        # ).astype(np.float32)
        node_features = np.zeros(
            (num_nodes, self.memory_dim),
            dtype=np.float32,
        )
        edge_features = np.zeros(
            (len(train_df) + 1, self.memory_dim), dtype=np.float32
        )
        edge_features[1:, 0] = 1.0
        node_features[0] = 0
        self.train_data = Data(
            sources=source_nodes,
            destinations=destination_nodes,
            timestamps=timestamps,
            edge_idxs=edge_idxs,
            labels=np.ones(len(train_df), dtype=np.float32),
        )
        self.train_neighbor_finder = get_neighbor_finder(
            self.train_data, uniform=False, max_node_idx=num_nodes - 1
        )

        mean_src, std_src, mean_dst, std_dst = compute_time_statistics(
            source_nodes, destination_nodes, timestamps
        )

        self.model = TGN(
            neighbor_finder=self.train_neighbor_finder,
            node_features=node_features,
            edge_features=edge_features,
            device=self.device,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            use_memory=self.use_memory,
            memory_update_at_start=self.memory_update_at_start,
            message_dimension=self.message_dim,
            memory_dimension=self.memory_dim,
            embedding_module_type=self.embedding_module,
            message_function=self.message_function,
            mean_time_shift_src=mean_src,
            std_time_shift_src=max(std_src, 1e-10),
            mean_time_shift_dst=mean_dst,
            std_time_shift_dst=max(std_dst, 1e-10),
            n_neighbors=self.n_degree,
            aggregator_type=self.aggregator,
            # aggregation_half_life=self.aggregation_half_life,
            memory_updater_type=self.memory_updater,
            use_edge_features=self.edge,
            trainable_node_features=self.trainable_node_features,
        ).to(self.device)
        if resume_epoch and checkpoint_path is not None:
            load_weights(self.model, checkpoint_path)
        self.train_edge_raw_features = self.model.edge_raw_features.clone()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        num_interactions = self.train_data.n_interactions
        num_memory_batches = int(np.ceil(num_interactions / self.memory_batch_size))
        num_batches = int(np.ceil(num_interactions / self.batch_size))
        all_item_nodes = (
            np.arange(
                self.n_users + 1,
                self.n_users + self.n_items + 1,
                dtype=np.int64,
            )
            if self.loss_type == "fce"
            else None
        )
        total_steps = max(1, (self.n_epochs - resume_epoch) * num_batches)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )
        patience_counter = 0

        for epoch in range(resume_epoch, self.n_epochs):
            self.model.train()
            if self.use_memory:
                self.model.memory.__init_memory__()
                with torch.no_grad():
                    self.model.memory.last_update.zero_()
            if self.loss_type == "bpr":
                self.seen_items_mask = np.zeros(
                    (self.n_users, self.n_items), dtype=bool
                )
            train_progress = tqdm(
                total=num_batches,
                desc=f"Train {epoch + 1}/{self.n_epochs}",
                leave=True,
            )
            epoch_loss = torch.zeros((), device=self.device)
            for memory_batch in range(num_memory_batches):
                accumulation_index = (
                    memory_batch % self.gradient_accumulation_steps
                )
                if accumulation_index == 0:
                    self.optimizer.zero_grad()
                    optimization_batch_start = ((memory_batch // self.gradient_accumulation_steps) * self.batch_size)
                    optimization_batch_size = min(self.batch_size, num_interactions - optimization_batch_start,)
                    optimization_batch_loss = torch.zeros((), device=self.device)

                start = memory_batch * self.memory_batch_size
                end = min(start + self.memory_batch_size, num_interactions)

                sources = self.train_data.sources[start:end]
                destinations = self.train_data.destinations[start:end]
                timestamps = self.train_data.timestamps[start:end]
                edge_idxs = self.train_data.edge_idxs[start:end]

                batch_size = end - start
                positive_item_ids = destinations - self.n_users - 1
                user_embeddings, positive_embeddings, memory, last_update = (
                    self.model.compute_event_embeddings(
                        sources,
                        destinations,
                        timestamps,
                        edge_idxs,
                        self.n_degree,
                    )
                )
                if self.loss_type == "fce":
                    catalog_logits = self._compute_full_catalog_logits(
                        user_embeddings,
                        memory,
                        last_update,
                        all_item_nodes,
                        timestamps,
                    )
                    targets = torch.as_tensor(
                        positive_item_ids, dtype=torch.long, device=self.device
                    )
                    loss = self.loss_function(catalog_logits, targets)
                else:
                    negative_item_ids = np.empty(
                        (batch_size, self.num_candidates),
                        dtype=np.int64,
                    )
                    if self.loss_type == "cse":
                        for row, positive_item_id in enumerate(
                            positive_item_ids
                        ):
                            sampled_ids = self.rng.choice(
                                self.n_items - 1,
                                size=self.num_candidates,
                                replace=False,
                            )
                            negative_item_ids[row] = sampled_ids + (
                                sampled_ids >= positive_item_id
                            )
                    else:
                        user_ids = sources - 1
                        valid_negative_rows = np.zeros(batch_size, dtype=bool)
                        for row in range(batch_size):
                            user_id = user_ids[row]
                            positive_item_id = positive_item_ids[row]
                            excluded = self.seen_items_mask[user_id].copy()
                            excluded[positive_item_id] = True
                            available = candidate_items[(first_seen[candidate_items] < timestamps[row]) & ~excluded[candidate_items]]
                            if len(available):
                                negative_item_ids[row] = self.rng.choice(available, size=self.num_candidates, replace=len(available) < self.num_candidates)
                                valid_negative_rows[row] = True
                            else:
                                negative_item_ids[row] = positive_item_id
                            self.seen_items_mask[user_id, positive_item_id] = True

                        self.seen_items_mask[user_ids, positive_item_ids] = True
                    negative_nodes = (
                        negative_item_ids + self.n_users + 1
                    ).reshape(-1)
                    negative_timestamps = np.repeat(
                        timestamps, self.num_candidates
                    )
                    negative_embeddings = (
                        self.model.compute_candidate_embeddings(
                            negative_nodes,
                            negative_timestamps,
                            memory,
                            last_update,
                            self.n_degree,
                        ).view(batch_size, self.num_candidates, -1)
                    )
                    positive_scores = torch.sum(
                        user_embeddings * positive_embeddings, dim=1
                    )
                    negative_scores = torch.einsum(
                        "bd,bnd->bn", user_embeddings, negative_embeddings
                    )
                    if self.loss_type == "bpr":
                        loss = self.loss_function(positive_scores[valid_negative_rows], negative_scores[valid_negative_rows]) if valid_negative_rows.any() else positive_scores.sum() * 0
                    else:
                        loss = self.loss_function(positive_scores, negative_scores)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                loss_weight = batch_size / optimization_batch_size
                (loss * loss_weight).backward()

                if self.use_memory:
                    self.model.memory.detach_memory()

                weighted_loss = loss.detach() * batch_size
                epoch_loss += weighted_loss
                optimization_batch_loss += weighted_loss
                is_last_memory_batch = (memory_batch + 1 == num_memory_batches)
                accumulation_complete = (accumulation_index + 1 == self.gradient_accumulation_steps)
                if accumulation_complete or is_last_memory_batch:
                    self.optimizer.step()
                    self.scheduler.step()
                    with torch.no_grad():
                        self.model.node_raw_features[0].zero_()
                    optimization_batch = (memory_batch // self.gradient_accumulation_steps)
                    train_progress.update(1)
                    if (optimization_batch % 50 == 0 or optimization_batch + 1 == num_batches):
                        mean_batch_loss = (optimization_batch_loss / optimization_batch_size).item()
                        train_progress.set_postfix(
                            loss=f"{mean_batch_loss:.4f}"
                        )
            train_progress.close()
            self.model.eval()
            mean_train_loss = (epoch_loss / num_interactions).item()
            log_dict = {"epoch": epoch + 1, "train/loss": mean_train_loss}
            if val_dataset is not None and metrics is not None:
                train_memory = self.model.memory.backup_memory() if self.use_memory else None
                val_predictions = self.predict(
                    val_dataset,
                    top_n=max_top_n,
                    split="validation",
                    desc=f"Validation {epoch + 1}/{self.n_epochs}",
                )
                holdout_users = val_dataset.get_holdout_users()
                val_metrics = metrics(val_predictions[holdout_users, :], val_dataset.get_holdout_array()[holdout_users])
                log_dict.update({f"val/{k}": float(v) for k, v in val_metrics.items()})

                val_ndcg = float(val_metrics["ndcg@20"])
                if val_ndcg > self.best_val_ndcg_20:
                    self.best_val_ndcg_20 = val_ndcg
                    self.best_epoch = epoch + 1
                    self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_train_memory = train_memory
                    patience_counter = 0
                else:
                    patience_counter += 1

                log_dict["val/patience"] = patience_counter

                if train_memory is not None:
                    self.model.memory.restore_memory(train_memory)
                self.model.set_neighbor_finder(self.train_neighbor_finder)
                self.model.edge_raw_features = self.train_edge_raw_features
                self.model.embedding_module.edge_features = self.train_edge_raw_features
                self.model.train()
            run.log(log_dict)
            if checkpoint_path is not None:
                save_weights(self.model, checkpoint_path)
            if (
                val_dataset is not None
                and metrics is not None
                and patience_counter >= self.patience
            ):
                break
        if val_dataset is not None and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            if self.best_train_memory is not None:
                self.model.memory.restore_memory(self.best_train_memory)
        if created_run:
            run.finish()
        
        return self

    @torch.no_grad()
    def predict(self, dataset, top_n, split=None, desc=None):
        self.model.eval()
        if split is None:
            split = "validation" if dataset._split == "val" else dataset._split
        if desc is None:
            desc = "Validation" if split == "validation" else "Test"

        train_df = dataset._df_train.copy()
        val_df = dataset._df_val.copy()

        train_df["split"] = "train"
        val_df["split"] = "validation"

        if split == "validation":
            history_df = pd.concat([train_df, val_df], ignore_index=True)
            seen_df = train_df
        else:
            test_df = dataset._df_test.copy()
            test_df["split"] = "test"
            history_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            seen_df = pd.concat([train_df, val_df], ignore_index=True)

        history_df = history_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        history_df["edge_idx"] = np.arange(1, len(history_df) + 1, dtype=np.int64)
        if not self.edge:
            history_df["edge_idx"] = 0
        events = history_df[history_df["split"] == split].reset_index(drop=True)

        targets = dataset._holdout_df.copy()
        predictions = np.zeros((self.n_users, top_n), dtype=np.int64)

        seen_items = [set() for _ in range(self.n_users)]

        for row in seen_df.itertuples(index=False):
            seen_items[int(row.user_id)].add(int(row.item_id))


        history_sources = history_df["user_id"].to_numpy(np.int64) + 1
        history_destinations = history_df["item_id"].to_numpy(np.int64) + self.n_users + 1
        history_timestamps = history_df["timestamp"].to_numpy(np.float64)
        history_edge_idxs = history_df["edge_idx"].to_numpy(np.int64)
        history_data = Data(sources=history_sources,
                            destinations=history_destinations,
                            timestamps=history_timestamps,
                            edge_idxs=history_edge_idxs,
                            labels=np.ones(len(history_df), dtype=np.float32))

        history_neighbor_finder = get_neighbor_finder(history_data, uniform=False)
        self.model.set_neighbor_finder(history_neighbor_finder)
        history_edge_features = np.zeros(
            (len(history_df) + 1, self.memory_dim), dtype=np.float32
        )
        history_edge_features[1:, 0] = 1.0
        history_edge_features = torch.from_numpy(history_edge_features).to(self.device)
        self.model.edge_raw_features = history_edge_features
        self.model.embedding_module.edge_features = history_edge_features
        target_keys = set(zip(targets["user_id"].astype(int), targets["item_id"].astype(int), targets["timestamp"]))
        all_items = np.arange(self.n_users + 1, self.n_users + self.n_items + 1, dtype=np.int64)


        def process_event_batch(batch):
            if not batch:
                return 
            users = np.fromiter((row[0] + 1 for row in batch), dtype=np.int64)
            items = np.fromiter((row[1] + self.n_users + 1 for row in batch), dtype=np.int64)
            times = np.fromiter((row[2] for row in batch), dtype=np.float64)
            edge_idxs = np.fromiter((row[3] for row in batch), dtype=np.int64)
            self.model.process_real_interaction(
                users, items, times, edge_idxs, self.n_degree
            )
            for user_id, item_id, _, _ in batch:
                seen_items[user_id].add(item_id)

        pending_events = []
        progress = tqdm(total=len(events), desc=desc, leave=True)
        for row in events.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)
            user = user_id + 1
            time = float(row.timestamp)
            edge_idx = int(row.edge_idx)
            is_target = (user_id, item_id, time) in target_keys
            user_array = np.array([user], dtype=np.int64)
            time_array = np.array([time], dtype=np.float64)

            if is_target:
                process_event_batch(pending_events)
                progress.update(len(pending_events))
                pending_events.clear()
                user_embedding, item_embeddings = self.model.compute_temporal_embeddings_read_only(user_array, all_items, time_array, self.n_degree)
                scores = torch.sum(user_embedding * item_embeddings, dim=1)

                true_item_score = scores[item_id].clone()

                if seen_items[user_id]:
                    seen = torch.tensor(list(seen_items[user_id]), dtype=torch.long, device=scores.device)
                    scores[seen] = -torch.inf

                scores[item_id] = true_item_score

                top_indices = torch.topk(scores, k=top_n).indices.cpu().numpy()
                predictions[user_id] = top_indices
                process_event_batch([(user_id, item_id, time, edge_idx)])
                progress.update(1)
            else:
                pending_events.append((user_id, item_id, time, edge_idx))
                if len(pending_events) >= self.batch_size:
                    process_event_batch(pending_events)
                    progress.update(len(pending_events))
                    pending_events.clear()

        process_event_batch(pending_events)
        progress.update(len(pending_events))
        progress.close()

        return predictions        

    
    def load_checkpoint(self, path):
        return super().load_checkpoint(path)
    def save_checkpoint(self, path):
        return super().save_checkpoint(path)
    def suggest_additional_params(self):
            return {"n_epochs": int(self.best_epoch)}
        
    
    
