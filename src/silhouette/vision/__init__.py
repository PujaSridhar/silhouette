"""W1: garment segmentation, and volume measured from a mask instead of judged by eye.

The engine's `volume` field is defined as garment width over body width at the
garment's own zone. In phase 0 that number is a judgement, typed into a CSV by
one person. A segmentation mask plus a known body scale turns it into a
measurement. That is what this package is for; reading photos is the means.

Layout:

    categories   Fashionpedia's ontology and how it maps onto the engine's
                 garment categories. Pure data.
    fashionpedia COCO-format loader: filtering, licence groups, mask decoding.
    volume       mask + body scale + frame -> measured volume, with warnings.
    metrics      per-class mask IoU.
    sensitivity  how many outfits change which rules fire when volume moves.
    volume_eval  measured vs hand-judged volume on her own photos.
    train        Mask2Former fine-tuning (needs the [vision] extra and a GPU).
    sam          SAM 2 as an annotation accelerator (needs the [sam] extra).

Everything except `train` and `sam` needs only numpy and Pillow, so the
measurement and the eval are testable without a GPU stack.
"""
