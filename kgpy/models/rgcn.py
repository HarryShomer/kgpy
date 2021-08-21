from typing import Optional, Union, Tuple


import torch, math
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul, masked_select_nnz

from .base_gnn_model import BaseGNNModel
# from torch_geometric.nn import MessagePassing
from .torch_message_passing import MessagePassing




def masked_edge_index(edge_index, edge_mask):
	if isinstance(edge_index, Tensor):
		return edge_index[:, edge_mask]
	else:
		return masked_select_nnz(edge_index, edge_mask, layout='coo')




class RGCN(BaseGNNModel):
	def __init__(self, 
			
			num_entities, 
			num_relations, 
			edge_index, 
			edge_type,
			gcn_dim = 200,
			emb_dim = 200,
			dropout=.1,
			regularization = None,
			reg_weight = 0,
			loss_fn="bce",
			device='cuda',
			num_layers=2,

			rgcn_num_bases=None,
			rgcn_num_blocks=100,

				):
		super(RGCN, self).__init__(
			type(self).__name__,
			edge_index = edge_index, 
			edge_type = edge_type,
			num_layers = num_layers,
			gcn_dim = gcn_dim,
			num_entities = num_entities, 
			num_relations = num_relations, 
			emb_dim = emb_dim,
			# loss_margin = 0, 
			# regularization = regularization, 
			# reg_weight =  reg_weight,
			weight_init = None, 
			loss_fn = loss_fn,
			# norm_constraint =  False,
			device=device
		)

		
		
		self.rgcn_num_bases = rgcn_num_bases
		self.rgcn_num_blocks = rgcn_num_blocks
		

		self.gcn_drop = torch.nn.Dropout(dropout)
		self.act	= torch.tanh

	
		self.entity_embeddings		= get_param((self.num_entities,   self.emb_dim), self.device)

		# self.init_rel = get_param(( edge_type.unique().size(0), self.p.embed_dim))

		self.relation_embeddings = get_param(( num_relations, self.emb_dim), self.device)
		self.w_rel 		= get_param((self.emb_dim, self.emb_dim), self.device)

		

		if self.num_layers == 1: self.act = None
		self.rgcn_conv1 = RGCNConv(self.emb_dim, self.gcn_dim, self.num_relations, self.rgcn_num_bases, self.rgcn_num_blocks, act=self.act, device=self.device)

		self.rgcn_conv2 = RGCNConv(self.gcn_dim, self.emb_dim, self.num_relations, self.rgcn_num_bases, self.rgcn_num_blocks, device=self.device) if self.num_layers == 2 else None

		
		self.register_parameter('bias', torch.nn.Parameter(torch.zeros(self.num_entities)))

	def forward(self,  triplets, **kwargs):
		
		

		x = self.rgcn_conv1(self.entity_embeddings, self.edge_index, self.edge_type)
		x = self.gcn_drop(x)

		x = self.rgcn_conv2(x, self.edge_index, self.edge_type) if self.num_layers == 2 else x
		x = self.gcn_drop(x) if self.num_layers == 2 else x

		
		r = torch.matmul(self.relation_embeddings, self.w_rel)
		
		
		if len(kwargs) == 0:  # train score
			sub_emb	= torch.index_select(x, 0, triplets[:, 0])
			rel_emb	= torch.index_select(r, 0, triplets[:, 1])
			obj_emb	= torch.index_select(x, 0, triplets[:, 2])

			score = self.get_train_score(sub_emb, rel_emb, obj_emb)
		else:     #test score
			
			sub_emb	= torch.index_select(x, 0, triplets[:, 1])
			rel_emb	= torch.index_select(r, 0, triplets[:, 0])

			score = self.get_test_score(sub_emb, rel_emb, x)


		return score


		
	def get_train_score(self, sub_emb, rel_emb, obj_emb):

		sr_emb	= sub_emb * rel_emb
		score = torch.sum(sr_emb* obj_emb, dim=1)
		
		
		
		return score

	def get_test_score(self, sub_emb, rel_emb, x):
		obj_emb	= sub_emb * rel_emb
		score = torch.mm(obj_emb, x.transpose(1, 0))

		
		
		return score

            




class RGCNConv(MessagePassing):

	def __init__(self, in_channels: int,
					out_channels: int,
					num_relations: int,
					num_bases: Optional[int] = None,
					num_blocks: Optional[int] = None,
					aggr: str = 'mean',
					root_weight: bool = True,
					bias: bool = True, 
					low_mem=False, 
					act=None,
					device='cuda',
					
					**kwargs):  # yapf: disable

		super(RGCNConv, self).__init__(aggr=aggr,  **kwargs)

		if num_bases is not None and num_blocks is not None:
			raise ValueError('Can not apply both basis-decomposition and '
								'block-diagonal-decomposition at the same time.')

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_relations = num_relations*2
		self.num_bases = num_bases
		self.num_blocks = num_blocks
		self.low_mem = low_mem
		self.act = act
		self.device = device

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)
		self.in_channels_l = in_channels[0]

		if num_bases is not None:
			# if num_bases > num_relations or num_bases <= 0:
			# 	self.num_bases = num_relations

			self.weight = get_param((self.num_bases, in_channels[0], out_channels),device=self.device)
			self.comp = get_param((self.num_relations, self.num_bases), device=self.device)
			

		elif num_blocks is not None:
			# if num_blocks > num_relations or num_blocks <= 0:
			# 	self.num_blocks = num_relations

			assert (in_channels[0] % self.num_blocks == 0
					and out_channels % self.num_blocks == 0)
			self.weight = get_param((self.num_relations, self.num_blocks,
								in_channels[0] // self.num_blocks,
								out_channels // self.num_blocks), device=self.device)
			self.register_parameter('comp', None)

		else:
			self.weight = get_param((self.num_relations, in_channels[0], out_channels), device=self.device)
			self.register_parameter('comp', None)

		if root_weight:
			self.root = get_param((in_channels[1], out_channels), device=self.device)
		else:
			self.register_parameter('root', None)

		if bias:
			
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
		else:
			self.register_parameter('bias', None)

		self.bn = torch.nn.BatchNorm1d(out_channels)


	


	def forward(self, x, edge_index, edge_type):
		

		if self.num_bases is not None:
			
			out = self.basic_decom_func(x, edge_index, edge_type)
		
		elif self.num_blocks is not None:
			
			out = self.block_decomp_func(x, edge_index, edge_type)
		
		else:
			raise ValueError('only implemnt RGCN with basic decompsition and block diagonal-decomposition')
		
		
		return out

	def basic_decom_func(self, x, edge_index, edge_type):
		

		if self.low_mem:
			weight = self.weight
		
			weight = (self.comp @ weight.view(self.num_bases, -1)).view(
			self.num_relations, self.in_channels_l, self.out_channels)
			out = torch.zeros(x.size(0), self.out_channels, device=x.device)

			for i in range(self.num_relations):
				tmp = masked_edge_index(edge_index, edge_type == i)

				h = self.propagate(tmp, x=x, edge_type=edge_type, norm=None)
				out = out + (h @ weight[i])
		else:
			norm = self.compute_norm(edge_type, edge_index, dim_size=x.size(0))
			self.aggr = 'add'
			out = self.propagate(edge_index, x=x, edge_type=edge_type, norm=norm)

		root = self.root
		if root is not None:
			out +=  x @ root

		if self.bias is not None:
			out += self.bias
		
		if self.bn is not None:
			out = self.bn(out)
		
		if self.act is not None:
			out = self.act(out)

		
		return out


	
	def block_decomp_func(self, x, edge_index, edge_type):
		

		if self.low_mem:
			weight = self.weight
			out = torch.zeros(x.size(0), self.out_channels, device=x.device)

			for i in range(self.num_relations):
				tmp = masked_edge_index(edge_index, edge_type == i)
				h = self.propagate(tmp, x=x, edge_type=edge_type, norm=None)
				h = h.view(-1, weight.size(1), weight.size(2))
				h = torch.einsum('abc,bcd->abd', h, weight[i])
				out += h.contiguous().view(-1, self.out_channels)
		
		else:
			norm = self.compute_norm(edge_type, edge_index, dim_size=x.size(0))
			self.aggr = 'add'
			
			out = self.propagate(edge_index, x=x, edge_type=edge_type, norm=norm)
			
		
		root = self.root
		if root is not None:
			out +=  x @ root
			
		if self.bias is not None:
			out += self.bias
		
		if self.bn is not None:
			out = self.bn(out)
		
		if self.act is not None:
			out = self.act(out)

		return out
		

	def message(self, x_j, edge_type, edge_index, norm):
		if self.low_mem:
			return x_j
		
		else:

			if  self.num_bases is not None:
				weight = self.weight
		
				weight = (self.comp @ weight.view(self.num_bases, -1)).view(
					self.num_relations, self.in_channels_l, self.out_channels)

				out = torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

				return out if norm is None else out * norm.view(-1, 1)
				
				

			elif self.num_blocks is not None:
				weight = self.weight
				
				weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
				x_j = x_j.view(-1, 1, weight.size(1))
				out = torch.bmm(x_j, weight).view(-1, self.out_channels)

				return out if norm is None else out * norm.view(-1, 1)


	def compute_norm(self,  edge_type: Tensor, index: Tensor,
					dim_size: Optional[int] = None) -> Tensor:

		# Compute normalization in separation for each `edge_type`.
		
		norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
		norm = scatter(norm, index[0], dim=0, dim_size=dim_size)[index[0]]
		norm = torch.gather(norm, 1, edge_type.view(-1, 1))
		norm = 1. / norm.clamp_(1.)	

		return norm
		# return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)	

def get_param(shape, device):
	param = torch.nn.Parameter(torch.Tensor(*shape).to(device))
	torch.nn.init.xavier_normal_(param.data)
	return param