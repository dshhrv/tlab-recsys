import torch
from torch import nn


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return
        last_updates = self.memory.get_last_update(unique_node_ids)
        assert (last_updates <= timestamps).all().item(), \
            "Trying to update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        with torch.no_grad():
            self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)
        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return (
                self.memory.get_all_memory(),
                self.memory.last_update.detach().clone()
            )

        last_updates = self.memory.get_last_update(unique_node_ids)
        assert (last_updates <= timestamps).all().item(), \
            "Trying to update memory to time in the past"

        # Build a differentiable preview of the GRU update.  The persistent
        # memory itself remains a non-gradient state buffer.
        updated_memory = self.memory.get_all_memory().clone()
        current_mem = updated_memory[unique_node_ids]
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_messages, current_mem
        )

        updated_last_update = self.memory.last_update.detach().clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        self.memory_updater = nn.RNNCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )


class LSTMMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(LSTMMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        self.memory_updater = nn.LSTMCell(
            input_size=message_dimension,
            hidden_size=memory_dimension
        )


class TransformerMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(TransformerMemoryUpdater, self).__init__(
            memory, message_dimension, memory_dimension, device
        )
        self.memory_updater = nn.TransformerEncoderLayer(
            d_model=message_dimension,
            nhead=4
        )


def get_memory_updater(
    module_type, memory, message_dimension, memory_dimension, device
):
    if module_type == "gru":
        return GRUMemoryUpdater(
            memory, message_dimension, memory_dimension, device
        )
    elif module_type == "rnn":
        return RNNMemoryUpdater(
            memory, message_dimension, memory_dimension, device
        )
    elif module_type == "lstm":
        return LSTMMemoryUpdater(
            memory, message_dimension, memory_dimension, device
        )
    elif module_type == "transformer":
        return TransformerMemoryUpdater(
            memory, message_dimension, memory_dimension, device
        )
    else:
        raise ValueError(f"Memory updater {module_type} not implemented")
