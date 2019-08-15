from fairseq.models.roberta import RobertaModel
import torch.nn.functional as F
import torch
import argparse
import numpy as np
def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None,type=str, required=True,
                        help="")
    parser.add_argument("--task", default=None,type=str, required=True,
                        help="")
    parser.add_argument("--data_dir", default=None,type=str, required=True,
                        help="")
    args = parser.parse_args()
    # print(args)

    roberta = RobertaModel.from_pretrained(
        args.output_dir, 
        # './outputs/RTE/7/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=args.data_dir
    )
    
    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )
    # print(label_fn)
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open('../data-superglue-csv/'+args.task+'/test.tsv') as fin:
        fin.readline()
        logits=np.array([])
        num_classes =2
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2 = tokens[0], tokens[1]
            tokens = roberta.encode(sent1, sent2)
            logit  = roberta.predict('sentence_classification_head', tokens).item()
            logits = np.append(logits,logit)
        print(logit)
        logits = logits.reshape((-1, num_classes))
        preds = np.argmax(logits, -1)

    print(preds)
 
    with open(args.output_dir+'pred_results', "w") as writer:
        # print(label_list)
        for i in range(len(preds)):
            # json_i= "\"idx: %d, \"label\": \"label_i\""
            writer.write("{\"idx\": %d, \"label\": \"%s\"}\n"%(i,preds[i]))
    # print(preds)


if __name__ == '__main__':

    predict()
