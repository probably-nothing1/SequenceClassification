from .SeqModel import GRU, LSTM

def dispatch_model(args, device):
    assert args.seq_model_type in ['lstm', 'gru']
    if args.seq_model_type == 'lstm':
        return LSTM(args.num_layers, args.hidden_size, embedding=args.embedding).to(device)
    elif args.seq_model_type == 'gru':
        return GRU(args.num_layers, args.hidden_size, embedding=args.embedding).to(device)