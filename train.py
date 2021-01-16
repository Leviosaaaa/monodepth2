# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
	if opts.HSV_mask:
		from hsv_trainer import hsv_Trainer
		trainer = hsv_Trainer(opts)
		print("Training with hsv_Trainer")
	else:
		trainer = Trainer(opts)
		print("Training with normal Trainer")
	trainer.train()
