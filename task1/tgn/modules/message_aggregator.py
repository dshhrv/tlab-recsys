import torch
import numpy as np
from collections import defaultdict



class MessageAggregator(torch.nn.Module):
    """
    Abstract class for the message aggregator module, which given a batch of node ids and
    corresponding messages, aggregates messages with the same node id.
    """

    def __init__(self, device):
        super(MessageAggregator, self).__init__()
        self.device = device

    def aggregate(self, node_ids, messages, memory=None):
        """
        Given a list of node ids, and a list of messages of the same length, aggregate different
        messages for the same id using one of the possible strategies.
        :param node_ids: A list of node ids of length batch_size
        :param messages: A dict where keys are node_ids and values are lists of (message, timestamp)
        :return: A tensor of aggregated messages
        """
        raise NotImplementedError

    def group_by_id(self, node_ids, messages, timestamps):
        node_id_to_messages = defaultdict(list)

        for i, node_id in enumerate(node_ids):
            node_id_to_messages[node_id].append((messages[i], timestamps[i]))

        return node_id_to_messages

class AttentionMessageAggregator(MessageAggregator):
    def __init__(self, memory_dim, message_dim, n_heads, device):
        super().__init__(device)

        self.query_projection = torch.nn.Linear(memory_dim, message_dim)
        self.scale = message_dim ** -0.5

    def aggregate(self, node_ids, messages, memory):
        unique_node_ids = np.unique(node_ids)
        to_update_node_ids = [node_id for node_id in unique_node_ids if messages[node_id]]

        if not to_update_node_ids:
            return [], [], []

        aggregated_messages = []
        unique_timestamps = []
        for node_id in to_update_node_ids:
            node_messages = messages[node_id]
            # clone() isolates the saved Linear input from in-place memory updates
            # that happen before loss.backward().
            q = self.query_projection(memory[node_id].detach().clone())
            stacked_messages = torch.stack([message for message, _ in node_messages])
            scores = torch.mv(stacked_messages, q) * self.scale
            alpha = torch.softmax(scores, dim=0)
            aggregated_messages.append(
                torch.sum(alpha.unsqueeze(1) * stacked_messages, dim=0)
            )
            unique_timestamps.append(node_messages[-1][1])

        return (
            to_update_node_ids,
            torch.stack(aggregated_messages),
            torch.stack(unique_timestamps),
        )
        

class LastMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(LastMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages, memory=None):
        """Only keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        if len(to_update_node_ids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = []
            unique_timestamps = []

        return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
    def __init__(self, device):
        super(MeanMessageAggregator, self).__init__(device)

    def aggregate(self, node_ids, messages, memory=None):
        """Compute the mean of all messages for each node"""
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                node_msgs = torch.stack([m[0] for m in messages[node_id]])
                unique_messages.append(torch.mean(node_msgs, dim=0))
                unique_timestamps.append(messages[node_id][-1][1])

        if len(to_update_node_ids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)
        else:
            unique_messages = []
            unique_timestamps = []

        return to_update_node_ids, unique_messages, unique_timestamps


class ExponentialMessageAggregator(MessageAggregator):
    def __init__(self, device, half_life):
        super().__init__(device)
        self.half_life = half_life

    def aggregate(self, node_ids, messages, memory=None):
        unique_node_ids = np.unique(node_ids)
        to_update_node_ids = []
        unique_messages = []
        unique_timestamps = []

        for node_id in unique_node_ids:
            node_messages = messages[node_id]
            if not node_messages:
                continue

            stacked_messages = torch.stack([message for message, _ in node_messages])
            timestamps = torch.stack([timestamp for _, timestamp in node_messages])
            latest_time = timestamps.max()
            log_weights = -(latest_time - timestamps) * (np.log(2) / self.half_life)
            weights = torch.softmax(log_weights, dim=0).to(
                device=stacked_messages.device, dtype=stacked_messages.dtype
            )

            to_update_node_ids.append(node_id)
            unique_messages.append(torch.sum(weights.unsqueeze(1) * stacked_messages, dim=0))
            unique_timestamps.append(latest_time.to(
                device=stacked_messages.device, dtype=torch.float32
            ))

        if not to_update_node_ids:
            return [], [], []

        return to_update_node_ids, torch.stack(unique_messages), torch.stack(unique_timestamps)


def get_message_aggregator(aggregator_type, device, memory_dim, message_dim, n_heads, half_life=600.0):
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    elif aggregator_type == "exponential":
        return ExponentialMessageAggregator(device=device, half_life=half_life)
    elif aggregator_type == "attention":
        return AttentionMessageAggregator(
            memory_dim=memory_dim,
            message_dim=message_dim,
            n_heads=n_heads,
            device=device
        )
    else:
        raise ValueError(f"Message aggregator {aggregator_type} not implemented")
