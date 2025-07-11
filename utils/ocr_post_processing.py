import numpy as np

def get_ignored_tokens():
    return [0]

def decode(character, text_index, text_prob=None, is_remove_duplicate=False):
    """将OCR输出的索引序列解码为字符串及置信度"""
    result_list = []
    ignored_tokens = get_ignored_tokens()
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token
        char_list = [
            character[int(text_id)].replace('\n', '')
            for text_id in text_index[batch_idx][selection]
        ]
        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]
        text = ''.join(char_list)
        result_list.append((text, np.mean(conf_list).tolist(), conf_list))
    return result_list