from .encoder_decoder import *

@SEGMENTORS.register_module(force=True)
class DepthEncoderDecoder(EncoderDecoder):
    def forward_train(self, img, img_metas, gt_depth):
        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_depth)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_depth)
            losses.update(loss_aux)

        return losses