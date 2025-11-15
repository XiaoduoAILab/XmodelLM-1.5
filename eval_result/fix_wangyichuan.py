import os
import json


json_dir = '/data/wangyichuan/llm_pretrain_xmodel/eval_result/c_line/medium'
to_dir = '/data/wangyichuan/llm_pretrain_xmodel/eval_result/c_line'
for file in os.listdir(json_dir):
    if file[:1] in ['a', 'b', 'o'] or file[:4] in ['piqa'] or file[:10] in ['winogrande']:
        with open(os.path.join(json_dir, file), 'r') as f:
            with open(os.path.join(to_dir, file), 'w') as fp:
                data = f.readlines()
                for index, line in enumerate(data):
                    temp = json.loads(line)
                    new = json.dumps(dict(ckp_name='iter-%06d'%(index*2000+2000), acc=temp['acc']))
                    fp.write(new + '\n')
