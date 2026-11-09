#!/usr/bin/env python3
"""dell_train_experts.py - DELL 复现: 9 折 x 7 专家 顺序训练驱动 (服务器端).

注意: 本脚本依赖 DELL 作者仓库的 `dataset`/`model` 模块 (经 EXPERT_DIR 的
sys.path.insert 导入), 且路径常量指向服务器环境. 仓库已附带该步产出的冻结
逐折专家预测 (dell_local/results/*.pt), 复现论文表格只需 dell_ensemble_qwen.py;
本脚本仅作为专家训练步骤的存档, 不参与 runs/ 一键链.

忠实移植 author train.py + inference.py:
  - MyModel = frozen DeBERTa-v3-large + GIN(2, 1024, dropout .5)
  - Adam lr 1e-4 wd 1e-5, batch 4, 100 epochs, early stop no_up>=16 (epoch>=8 起), val=macro-F1
  - 差异(披露): 每 (fold, expert) 只训 1 个种子 (作者 5 个), 其余全部一致

7 专家 (vanilla.py 的 ensemble 列表):
  text: None / emotion / framing / propaganda / retrieval (graph=None)
  graph: stance / relation (text=None)

用法:
  python dell_train_experts.py --folds 0 1 2 3 4 5 6 7 8 --experts all
  python dell_train_experts.py --folds 0 --experts None_None emotion_None  # 断点续跑自动跳过
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

TASK = 'TASK1'
DATA_ROOT = '/home/user1809/wenyang/DELL/data'
EXPERT_DIR = '/home/user1809/wenyang/DELL/expert'
RESULT_DIR = '/home/user1809/wenyang/DELL/expert/results'

EXPERTS = [  # (text_augmentation, graph_augmentation) — author 的 key 名
    ('None', 'None'),
    ('emotion', 'None'),
    ('framing', 'None'),
    ('propaganda', 'None'),
    ('retrieval', 'None'),
    ('None', 'stance'),
    ('None', 'relation'),
]

sys.path.insert(0, EXPERT_DIR)
from dataset import NewsDataset, my_collate_fn, MySampler  # noqa: E402
from model import MyModel  # noqa: E402

device = torch.device('cuda')
LM_PATH = os.environ.get('DELL_LM_PATH', 'microsoft/deberta-v3-large')


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_truth, all_preds, save_out = [], [], []
    for batch in loader:
        out, _ = model(batch)
        preds = out.argmax(-1).to('cpu')
        all_truth.append(batch['label'].to('cpu'))
        all_preds.append(preds)
        save_out.append(out.to('cpu'))
    return (get_metric(torch.cat(all_truth).numpy(), torch.cat(all_preds).numpy()),
            torch.cat(save_out), torch.cat(all_preds), torch.cat(all_truth))


def train_one_fold_expert(fold, text_aug, graph_aug, log):
    ds_name = f'PHEME18_f{fold}'
    name = f'{TASK}_{ds_name}_{text_aug}_{graph_aug}'
    ckpt_dir = f'{EXPERT_DIR}/checkpoints/{name}'
    marker = f'{RESULT_DIR}/{name}_test.pt'
    if os.path.exists(marker):
        print(f'[skip] {name}', flush=True)
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f'== [{name}] loading dataset...', flush=True)
    dataset = NewsDataset(TASK, ds_name, root=DATA_ROOT)
    print(f'== dataset loaded ({len(dataset)} posts), building loaders', flush=True)
    train_idx = json.load(open(f'{DATA_ROOT}/split/{TASK}_{ds_name}/train.json'))
    val_idx = json.load(open(f'{DATA_ROOT}/split/{TASK}_{ds_name}/val.json'))
    test_idx = json.load(open(f'{DATA_ROOT}/split/{TASK}_{ds_name}/test.json'))
    loaders = {}
    for split, idx, shuf in (('train', train_idx, True), ('val', val_idx, False), ('test', test_idx, False)):
        loaders[split] = DataLoader(dataset, batch_size=4, collate_fn=my_collate_fn,
                                    sampler=MySampler(idx, shuffle=shuf))

    model = MyModel(num_class=2, task=TASK, device=device,
                    lm_path=LM_PATH,
                    text_augmentation=None if text_aug == 'None' else text_aug,
                    graph_augmentation=None if graph_aug == 'None' else graph_aug).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    print('== model on GPU, training starts', flush=True)

    best_acc, best_state, no_up = 0, None, 0
    t0 = time.time()
    for epoch in range(100):
        model.train()
        for batch in loaders['train']:
            optimizer.zero_grad()
            _, loss = model(batch)
            loss.backward()
            optimizer.step()
        acc, _, _, _ = evaluate(model, loaders['val'])
        if acc > best_acc:
            best_acc = acc
            no_up = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            if epoch >= 8:
                no_up += 1
        if no_up >= 16:
            break
        print(f'  [{name}] ep{epoch} val={acc:.2f} best={best_acc:.2f} '
              f'({time.time()-t0:.0f}s)', flush=True)
    model.load_state_dict(best_state)
    test_metric, out, pred, label = evaluate(model, loaders['test'])
    torch.save([out, pred, label], marker)
    _, vout, vpred, vlabel = evaluate(model, loaders['val'])
    torch.save([vout, vpred, vlabel], f'{RESULT_DIR}/{name}_val.pt')
    log.write(json.dumps({'name': name, 'best_val': best_acc,
                          'test_macro_f1': test_metric,
                          'epochs': epoch + 1, 'secs': time.time() - t0}) + '\n')
    log.flush()
    print(f'[done] {name} val={best_acc:.2f} test={test_metric:.2f} '
          f'ep={epoch+1} ({time.time()-t0:.0f}s)', flush=True)
    del model, best_state
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(9)))
    ap.add_argument('--experts', type=str, nargs='+', default=['all'])
    args = ap.parse_args()
    exps = EXPERTS if args.experts == ['all'] else [tuple(e.split('_')) for e in args.experts]

    with open('/home/user1809/wenyang/DELL/dell_train_log.jsonl', 'a') as log:
        for fold in args.folds:
            for text_aug, graph_aug in exps:
                train_one_fold_expert(fold, text_aug, graph_aug, log)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
