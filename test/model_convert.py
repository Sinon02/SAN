import numpy as np
import torch
import paddle


def torch2paddle():
    torch_path = "./checkpoints/SAN_decoder/best.pth"
    paddle_path = "./checkpoints/SAN_decoder/best_paddle.pdparams"
    torch_state_dict = torch.load(torch_path, map_location=torch.device('cpu'))
    model_state_dict = torch_state_dict['model']
    fc_names = [
        "hidden_weight",
        "attention_weight",
        'alpha_convert',
        "init_weight",
        'word_state_weight',
        'word_convert',
        'struct_convert',
        'c2p_state_weight',
        'c2p_word_weight',
        'c2p_relation_weight',
        'c2p_context_weight',
        'c2p_convert',
        'word_embedding_weight',
        'word_context_weight'
    ]
    paddle_state_dict = {}
    for k in model_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = model_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k and len(v.shape) > 1:  # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        if False:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
