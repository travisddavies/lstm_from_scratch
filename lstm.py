import torch
import torch.nn as nn


class NativeCustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, device):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.device = device
        # f_t
        # These are the forget gate parameters
        # This parameter is for the input x_t
        self.U_f = nn.Parameter(torch.Tensor((input_sz, hidden_sz)))
        # This parameter is for the previous hidden state
        self.V_f = nn.Parameter(torch.Tensor((hidden_sz, hidden_sz)))
        # This is the bias term for the forget gate
        self.b_f = nn.Parameter(torch.Tensor((hidden_sz)))
        # i_t
        # This acts as the input gate
        # This parameter is for the input x_t
        self.U_i = nn.Parameter(torch.Tensor((input_sz, hidden_sz)))
        # This parameter is for the previous hidden state
        self.V_i = nn.Parameter(torch.Tensor((hidden_sz, hidden_sz)))
        # This is the bias term for the forget gate
        self.b_i = nn.Parameter(torch.Tensor((hidden_sz)))
        # c_t
        # This acts as the candidates
        # This parameter is for the input x_t
        self.U_c = nn.Parameter(torch.Tensor((input_sz, hidden_sz)))
        # This parameter is for the previous hidden state
        self.V_c = nn.Parameter(torch.Tensor((hidden_sz, hidden_sz)))
        # This is the bias term for the forget gate
        self.b_c = nn.Parameter(torch.Tensor((hidden_sz)))
        # o_t
        # This acts as the output gate
        # This parameter is for the input x_t
        self.U_o = nn.Parameter(torch.Tensor((input_sz, hidden_sz)))
        # This parameter is for the previous hidden state
        self.V_o = nn.Parameter(torch.Tensor((hidden_sz, hidden_sz)))
        # This is the bias term for the forget gate
        self.b_o = nn.Parameter(torch.Tensor((hidden_sz)))

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(self.device)
            c_t = torch.zeros(bs, self.hidden_size).to(self.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # Reshape hidden seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)
