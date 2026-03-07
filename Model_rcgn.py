import os
import gc

# --- FIX 1 & 2: Enable CPU Fallback & Disable Memory Limit ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# This allows using more than the "safe" 9GB limit (use with caution)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
import numpy as np

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"🔹 Using MPS device for Mac M2/M3: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"🔹 Using CPU device: {DEVICE}")

# --- MODEL DEFINITIONS ---

class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        # AGGRESSIVE OPTIMIZATION: Reduced num_bases from 30 to 4
        # This significantly lowers parameter memory
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=4).to(DEVICE)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=4).to(DEVICE)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, embedding_dim, num_relations):
        super().__init__()
        self.relation_embed = Parameter(torch.Tensor(num_relations, embedding_dim)).to(DEVICE)
        torch.nn.init.xavier_uniform_(self.relation_embed)

    def forward(self, h_emb, t_emb, r_type):
        r_emb = self.relation_embed[r_type]
        score = torch.sum(h_emb * r_emb * t_emb, dim=1)
        return score

# --- TRAINING LOGIC ---

def train_epoch(loader, features_cpu, encoder, decoder, optimizer, criterion, 
                full_edge_index, full_edge_type, train_edge_idx_np):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    num_train_links = len(train_edge_idx_np)
    
    # Reduced Link Batch Size (must match/be smaller than Node Batch)
    BATCH_SIZE_LINKS = 1024 
    
    for i, batch in enumerate(loader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        # Slice features on CPU, then move to GPU
        batch_features = features_cpu[batch.n_id.cpu()].to(DEVICE)

        h = encoder(batch_features, batch.edge_index, batch.edge_type)
        
        # --- Manual Link Batching ---
        pos_idx_sample = np.random.choice(train_edge_idx_np, size=BATCH_SIZE_LINKS, replace=False)
        pos_idx_sample = torch.tensor(pos_idx_sample, dtype=torch.long, device=DEVICE)
        
        pos_edge_index = full_edge_index[:, pos_idx_sample]
        pos_edge_type = full_edge_type[pos_idx_sample]
        
        neg_edge_index = negative_sampling(
            pos_edge_index, 
            num_nodes=features_cpu.size(0), 
            num_neg_samples=1
        )
        
        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        
        pos_labels = torch.ones(BATCH_SIZE_LINKS, dtype=torch.float, device=DEVICE)
        neg_labels = torch.zeros(BATCH_SIZE_LINKS, dtype=torch.float, device=DEVICE)
        target = torch.cat([pos_labels, neg_labels])
        
        all_r_type = torch.cat([pos_edge_type, pos_edge_type])
        
        node_id_to_local_idx = {global_id.item(): local_idx for local_idx, global_id in enumerate(batch.n_id)}

        try:
            h_indices_list = [node_id_to_local_idx.get(i.item()) for i in all_edge_index[0]]
            t_indices_list = [node_id_to_local_idx.get(i.item()) for i in all_edge_index[1]]
            
            valid_mask = [h is not None and t is not None for h, t in zip(h_indices_list, t_indices_list)]
            
            if not any(valid_mask):
                continue

            valid_mask_tensor = torch.tensor(valid_mask, device=DEVICE)
            h_indices = torch.tensor([h for h in h_indices_list if h is not None], device=DEVICE)
            t_indices = torch.tensor([t for t in t_indices_list if t is not None], device=DEVICE)
            
            target = target[valid_mask_tensor]
            all_r_type = all_r_type[valid_mask_tensor]

            h_emb = h[h_indices]
            t_emb = h[t_indices]

            pred_score = decoder(h_emb, t_emb, all_r_type)
            loss = criterion(pred_score, target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            
            # Explicit Cleanup
            del batch, h, h_emb, t_emb, pred_score, loss
            
        except Exception as e:
            print(f"Skipping batch due to error: {e}")
            continue

    return total_loss / (num_train_links * 2) 

# --- MAIN EXECUTION ---

def run_training():
    print("\n--- 1. Data and Feature Initialization ---")
    
    try:
        data_dict = torch.load('graph_data.pt', map_location='cpu')
        num_nodes = data_dict['num_nodes']
        num_relations = data_dict['num_relations']
        
        full_edge_index = data_dict['edge_index'].to(DEVICE)
        full_edge_type = data_dict['edge_type'].to(DEVICE)
        train_edge_idx_np = data_dict['train_idx'].numpy()
        
    except FileNotFoundError:
        print("CRITICAL ERROR: 'graph_data.pt' not found.")
        return

    # --- SIMULATE FEATURE LOADING ---
    EMBEDDING_DIM_BERT = 768  
    EMBEDDING_DIM_PBG = 100   
    
    print(f"Loading simulated features...")
    # Keep on CPU
    bert_features = torch.randn(num_nodes, EMBEDDING_DIM_BERT)
    pbg_features = torch.randn(num_nodes, EMBEDDING_DIM_PBG)
    initial_features_cpu = torch.cat([bert_features, pbg_features], dim=1)
    IN_CHANNELS = initial_features_cpu.size(1)

    print(f"✅ Data loaded. Nodes: {num_nodes:,}, Relations: {num_relations}")
    print(f"✅ Features kept on CPU. Dimension: {IN_CHANNELS}")
    
    full_graph = Data(
        x=initial_features_cpu, 
        edge_index=full_edge_index, 
        edge_type=full_edge_type,
        num_nodes=num_nodes
    )
    
    # --- AGGRESSIVE MEMORY OPTIMIZATION CONFIG ---
    HIDDEN_CHANNELS = 64  # Reduced from 128
    OUT_CHANNELS = 32     # Reduced from 64
    NUM_EPOCHS = 50 
    NODE_SAMPLE_BATCH_SIZE = 128 # Reduced from 512
    
    encoder = RGCNEncoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, num_relations)
    decoder = LinkPredictor(OUT_CHANNELS, num_relations)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.005
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Initializing NeighborLoader (Batch Size: {NODE_SAMPLE_BATCH_SIZE})...")
    
    train_loader = NeighborLoader(
        data=full_graph,
        # AGGRESSIVE REDUCTION: Only 10 neighbors in hop 1, 5 in hop 2
        num_neighbors=[10, 5], 
        input_nodes=torch.arange(num_nodes),
        batch_size=NODE_SAMPLE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"\n--- 2. Starting RGCN Training (Optimized for M2 Memory) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(
            train_loader, 
            initial_features_cpu,
            encoder, 
            decoder, 
            optimizer, 
            criterion, 
            full_edge_index, 
            full_edge_type, 
            train_edge_idx_np
        )
        
        # Force Memory Cleanup after each epoch
        torch.mps.empty_cache()
        gc.collect()
        
        print(f'Epoch: {epoch:03d}/{NUM_EPOCHS:03d}, Avg Loss: {loss:.4f}')
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, 'rgcn_link_predictor_final.pt')
    print("\n✅ Training complete.")

if __name__ == "__main__":
    run_training()