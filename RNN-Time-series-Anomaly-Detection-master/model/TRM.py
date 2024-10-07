import torch
import torch.nn as nn
from torch.autograd import Variable
import shutil
from pathlib import Path
class TransformerPredictor(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, enc_inp_size, trm_inp_size, trm_hid_size, dec_out_size, nhead, num_encoder_layers, dropout=0.5, tie_weights=False, res_connection=False):
        super(TransformerPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(enc_inp_size, trm_inp_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trm_inp_size, nhead=nhead, dim_feedforward=trm_hid_size, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(trm_inp_size, dec_out_size)

        if tie_weights:
            if trm_inp_size != dec_out_size:
                raise ValueError('When using the tied flag, trm_inp_size must be equal to dec_out_size')
            self.decoder.weight = self.encoder.weight

        self.res_connection = res_connection
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, return_hiddens=False):
        emb = self.drop(self.encoder(input))  # [seq_len, batch_size, feature_size]
        emb = emb.permute(1, 0, 2)  # Transformer expects [batch_size, seq_len, feature_size]
        output = self.transformer(emb)  # [batch_size, seq_len, feature_size]
        output = output.permute(1, 0, 2)  # Back to [seq_len, batch_size, feature_size]
        decoded = self.decoder(output)  # [seq_len, batch_size, dec_out_size]

        if self.res_connection:
            decoded = decoded + input

        if return_hiddens:
            return decoded, output

        return decoded

    def save_checkpoint(self, state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path('save', args.data, 'checkpoint')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')

        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save', args.data, 'model_best')
            model_best_dir.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

        print('=> checkpoint saved.')



    def initialize(self, args, feature_dim):
        self.__init__(
            enc_inp_size=feature_dim,
            trm_inp_size=args.emsize,
            trm_hid_size=args.nhid,
            dec_out_size=feature_dim,
            nhead=args.nhead,
            num_encoder_layers=args.nlayers,
            dropout=args.dropout,
            tie_weights=args.tied,
            res_connection=args.res_connection
        )
        self.to(args.device)

    def load_checkpoint(self, args, checkpoint, feature_dim):
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size = args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss


