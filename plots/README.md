1. 2D visualization of learned embedding vectors. 
    1.1 2D_EmbVec
    - Embedding vectors with accuracy 0.95, taken from hardy-sweep-2
    - Use PCA and then T-SNE to reduce the learned vectors to 2D space
    - Find QMOFs with same "decorated_graph_hash" and highlighted the pairs in the scatter plot
    - MOF pairs in this plot:
        [['qmof-1ecf90e', 'qmof-543c417'],
        ['qmof-198e8dc', 'qmof-7bda122'],
        ['qmof-62cf3da', 'qmof-b2f2b6c'],
        ['qmof-876addd', 'qmof-3a3a471'],
        ['qmof-da0a751', 'qmof-33487ab']]
    1.2 2D_EmbVec_RSM
    - Embedding vectors taken from crisp-sweep-42, with accuracy 0.94
    - There are only RSM structures
    - "undecorated_graph_hash"

2. infer_stability
    - Basic model configurations:
        - model training epochs=10
        - RSM
    - Descriptions:
        - v0 = model.dv
        - v1 = model.infer_vector
        - v2 = model.infer_vector
        - upper: the cos(v0, v1) distribution with different inference epochs 10(blue), 40(orange), 100(green)
        - lower: the cos(v1, v2) distribution with different inference epochs 10, 40, 100

3. train_infer_epochs
    - To check the optimal training and inference epochs
    - The training and inference epochs are shown in the plot
    - Trained on RSM database and infer also RSM db
