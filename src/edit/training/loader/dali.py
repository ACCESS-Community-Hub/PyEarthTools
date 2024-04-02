# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

# import torch
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
# from nvidia.dali import pipeline_def, Pipeline
# from typing import Union


# from edit.pipeline.templates import DataStep, DataIterator
# from edit.pipeline.sequential import Sequential, SequentialDecorator


# def DALIPipeline()

# @SequentialDecorator
# class DALILoader(DataStep):
#     def __init__(self, index: DataStep | DataIterator) -> None:
#         super().__init__(index = index)

#     def setup(self, stage=None):
#         device_id = self.local_rank
#         shard_id = self.global_rank
#         num_shards = self.trainer.world_size
#         mnist_pipeline = GetMnistPipeline(batch_size=BATCH_SIZE, device='gpu', device_id=device_id, shard_id=shard_id, num_shards=num_shards, num_threads=8)

#         class LightningWrapper(DALIGenericIterator):
#             def __init__(self, *kargs, **kvargs):
#                 super().__init__(*kargs, **kvargs)

#             def __next__(self):
#                 out = super().__next__()
#                 # DDP is used so only one pipeline per process
#                 # also we need to transform dict returned by DALIClassificationIterator to iterable
#                 # and squeeze the lables
#                 out = out[0]
#                 return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

#         self.train_loader = LightningWrapper(mnist_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

#     def train_dataloader(self):
#         return self.train_loader
