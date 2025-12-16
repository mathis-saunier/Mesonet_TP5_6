
class ViT(nn.Module):
    def __init__(self, config, device):
        super(ViT, self).__init__()
        self.input_features = config['input_features']
        self.batch_size = config['batch_size']
        self.num_epochs =config['num_epochs']
        self.learning_rate =config['learning_rate']
        self.num_classes =config['num_classes']
        self.hidden_size =config['hidden_size']
        self.num_heads =config['num_heads']
        self.num_layers =config['num_layers']
        self.dropout =config['dropout']
        self.x_pad_idx =config['x_pad_idx']
        self.y_pad_idx =config['y_pad_idx']
        self.max_length =config['max_length']
        self.START_TOKEN =config['START_TOKEN']
        self.END_TOKEN =config['END_TOKEN']
        self.DEVICE = device
        
        self.w_width = config['w_width']
        self.src_grid_size = 120 // self.w_width

        self.positional_encoding_layer = PositionalEncoding(dim_model = self.hidden_size, 
                                                            dropout_p = self.dropout, 
                                                            max_len = self.max_length)
        
        self.positional_encoding_layer2D = PositionalEncoding2D(dim_model = self.hidden_size, 
                                                            dropout_p = self.dropout, 
                                                            max_len = self.src_grid_size)
        
        self.x_embedding = nn.Linear(config['input_features'],config['hidden_size'])      
                
        self.y_embedding = nn.Embedding(config['num_classes'],config['hidden_size'])         
        
        self.transformer = nn.Transformer(d_model=config['hidden_size'], 
                                                      nhead=config['num_heads'], 
                                                      num_encoder_layers =config['num_layers'],
                                                      num_decoder_layers =config['num_layers'],
                                                      dim_feedforward = config['hidden_size'],
                                                      dropout=config['dropout'],
                                                      batch_first=True)
        self.output_layer = nn.Linear(config['hidden_size'], config['num_classes'])
        
    def forward(self, x,y,y_output_mask,src_key_padding_mask,tgt_key_padding_mask):
        x = self.x_embedding(x)
        x = self.positional_encoding_layer2D(x)
        
        y = self.y_embedding(y) * math.sqrt(self.hidden_size)
        y = self.positional_encoding_layer(y)

        transformer_out = self.transformer(x, y,
                                           tgt_mask = y_output_mask,
                                           src_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.output_layer(transformer_out)
        return output
    
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
