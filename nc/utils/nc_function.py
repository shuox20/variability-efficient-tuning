import torch

#final version in paper
def compute_nc(vectors):
    m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()], 1)
    B = torch.cov(m_all, correction=0)
    W = torch.zeros(B.shape)
    for v in vectors.values():
        W = W + torch.cov(v.transpose(0,1), correction=0)
    W = W/len(vectors)
    nc = 1/len(vectors)*torch.trace(torch.matmul(W, torch.linalg.pinv(B)))
    return nc

# compare different ways to compute mean and variance
def compare_nc(vectors):
    m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()], 1)
    B_2 = torch.cov(m_all, correction=0)
    m_global = torch.mean(torch.concat([v for v in vectors.values()], 0), 0, keepdims=True).transpose(0,1)
    B_1 = torch.mm(m_all - m_global, (m_all - m_global).transpose(0,1))/len(vectors)
    weights = torch.tensor([vector.shape[0] for vector in vectors.values()])
    W_2 = torch.zeros(B_2.shape)
    W_1 = torch.zeros(B_2.shape)
    for v in vectors.values():
        W_2 = W_2 + torch.cov(v.transpose(0,1), correction=0)
        W_1 = W_1 + torch.cov(v.transpose(0,1), correction=0)*v.shape[0]
    W_2 = W_2/len(vectors)
    W_1 = W_1/torch.sum(weights)
    B_3 = torch.cov(m_all, correction=0, fweights = weights)
    B_tmp3 = torch.zeros(B_2.shape)
    B_tmp4 = torch.zeros(B_2.shape)
    for i in range(len(vectors)):
        B_tmp4 = B_tmp4 + torch.mm((m_all[:,i].reshape(-1,1) - m_global), (m_all[:,i].reshape(-1,1) - m_global).transpose(0,1))
    B_tmp4 = B_tmp4/len(vectors)
    nc_balanced = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_1)))
    nc_imbalanced = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_2)))
    nc_3 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_2)))
    nc_4 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_3)))
    nc_5 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_1)))
    nc_6 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_3)))
    nc_7 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_tmp4)))
    nc_8 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_tmp4)))

    return([nc_balanced,nc_imbalanced, nc_3, nc_4, nc_5, nc_6, nc_7, nc_8])