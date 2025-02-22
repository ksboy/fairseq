# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
from sklearn.metrics import f1_score


@register_criterion('multiple_choice')
class MultipleChoiceCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
               'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=multiple_choice"

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name='sentence_classification_head',
        )

        num_classes = self.args.num_classes
        # print(logits)

        logits = torch.reshape(logits, (-1, num_classes))
        # print(logits)
        _, preds = torch.max(logits, -1)
        # print(preds)

        targets = model.get_targets(sample, [logits]).view(-1)
        # print(targets)
        sample_size = targets.numel()
        targets = torch.reshape(targets, (-1, num_classes))
        # print(targets)
        _, targets = torch.min(targets, -1)
        # print(targets)

        if not self.args.regression_target:
            loss = F.cross_entropy(
                logits,
                targets,
                reduction='sum',
            )
            # loss = F.nll_loss(
            #     F.log_softmax(logits, dim=-1, dtype=torch.float32),
            #     targets,
            #     reduction='sum',
            # )
        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': int(sample_size / num_classes),
            'sample_size': sample_size,
        }

        # print(preds, targets)
        if not self.args.regression_target:
            logging_output.update(
                ncorrect=(preds == targets).sum().item(),
                preds=preds,
                targets=targets
            )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # print(tuple(log.get('preds', 0) for log in logging_outputs))
        preds = torch.cat(tuple(log.get('preds', 0) for log in logging_outputs), 0)
        targets = torch.cat(tuple(log.get('targets', 0) for log in logging_outputs), 0)
        # print(preds, targets)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        agg_output.update(preds=preds.cpu())
        agg_output.update(targets=targets.cpu())
        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            # print(ncorrect, nsentences)
            agg_output.update(accuracy=ncorrect / nsentences)
            agg_output.update(f1=f1_score(targets.cpu(), preds.cpu(), average='macro'))

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
