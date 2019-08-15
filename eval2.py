from fairseq.models.roberta import RobertaModel
import torch
import argparse

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
    with open('../data-superglue-csv/'+args.task+'/val.tsv') as fin:
        fin.readline()
        preds=[]
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            # print(tokens)
            sent1, sent2 = tokens[0], tokens[1]
            # print(sent1,"\n", sent2)
            tokens = roberta.encode(sent1, sent2)
            # print(tokens)
            if len(tokens)>512:
                # print(len(tokens))
                # print(tokens)
                tokens= torch.cat((tokens[0].reshape(1),tokens[-511:]),0)
                # print(tokens)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            # print(prediction)
            prediction_label = label_fn(prediction)
            # print(prediction_label)
            preds.append(prediction_label )

    with open(args.output_dir+'eval_results2', "w") as writer:
        # print(label_list)
        for i in range(len(preds)):
            # json_i= "\"idx: %d, \"label\": \"label_i\""
            writer.write("{\"idx\": %d, \"label\": \"%s\"}\n"%(i,preds[i]))
    # print(preds)


if __name__ == '__main__':

    predict()
