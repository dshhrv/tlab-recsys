import torch
from torch import nn
from collections import defaultdict


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension,
                 message_dimension=None, device="cpu", combination_method='sum'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.combination_method = combination_method
        self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                        requires_grad=False)

        self.messages = defaultdict(list)

    def _node_indices(self, node_idxs):
        return torch.as_tensor(
            node_idxs, dtype=torch.long, device=self.memory.device
        )

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        node_idxs = self._node_indices(node_idxs)
        return self.memory[node_idxs, :]

    def get_all_memory(self):
        return self.memory

    def set_memory(self, node_idxs, values):
        node_idxs = self._node_indices(node_idxs)
        with torch.no_grad():
            self.memory[node_idxs, :] = values.detach()

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return (
            self.memory.detach().clone(),
            self.last_update.detach().clone(),
            messages_clone
        )

    def restore_memory(self, memory_backup):
        with torch.no_grad():
            self.memory.copy_(memory_backup[0])
            self.last_update.copy_(memory_backup[1])

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))
            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []



class TrainableMemory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, device="cpu"):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device
        self.memory = nn.Parameter(
            torch.empty(n_nodes, memory_dimension, device=device)
        )
        nn.init.xavier_uniform_(self.memory)
        with torch.no_grad():
            self.memory[0].zero_()

        self.last_update = torch.nn.Parameter(
            torch.zeros(n_nodes, dtype=torch.long, device=device),
            requires_grad=False
        )
        # self.messages = defaultdict(list)

    def __init__memory(self):
        with torch.no_grad():
            self.last_update.zero_()
        # self.messages = defaultdict(list)

    def _node_indices(self, node_idxs):
        return torch.as_tensor(
            node_idxs, dtype=torch.long, device=self.memory.device
        )

    def get_memory(self, node_idxs):
        node_idxs = self._node_indices(node_idxs)
        return self.memory[node_idxs, :]

    def get_all_memory(self):
        return self.memory

    def get_last_update(self, node_idxs):
        node_idxs = self._node_indices(node_idxs)
        return self.last_update[node_idxs]

    def set_memory(self, node_idxs, values):
        pass

    def store_raw_messages(self, nodes, node_id_to_messages):
        pass

    def backup_memory(self):
        return self.last_update.detach().clone()

    def restore_memory(self, memory_backup):
        if memory_backup is not None:
            with torch.no_grad():
                self.last_update.copy_(memory_backup)

    def detach_memory(self):
        pass

    def clear_messages(self, nodes):
        pass
