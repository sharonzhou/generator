

class Evaluator(object):
    # TODO: move loss evaluation and logger calls here
    """Evaluator class to perform two evaluations:
        (1) Mask: Evaluate masked, obscured, and full losses
        (2) Z-test: Backprop Z vector to held out test image(s)
    """
    def __init__(self, logger, use_intermediate_logits, **kwargs):
        self.logger = logger
        self.use_intermediate_logits = args.use_intermediate_logits
   
    def _compute_loss(self, loss):
        return

    def evaluate_mask(self, losses):
        for loss in losses:
            self._compute_loss(loss)
            logger.log()
        logger.visualize()
        return

    def evaluate_z_test(self, curr_model, img_test=None, **kwargs):
        # TODO: incl param that checks whether to run this test - prolly every n steps
        if z_test is None:
            # Sample random z vector, same shape as input
            # OK if same as input b/c on held out test img
            util.create_input_vec_thing()

        if img_test is None:
            # Get test image(s)
            # Ensure not the same as any in training

        # Make deep copy of model; get layers list with weights
        model = curr_model.deepcopy()
        # Prepend z vec to model beginning in module list, and make only that layer trainable
        # https://discuss.pytorch.org/t/is-there-a-way-to-prepend-a-module-to-the-beginning-or-a-modulelist/22688
        
        model.train()
        test_logger.init # TODO: need to some kind of loop/iters for this
        # TODO: get same optimizer as train used
        while not test_logger.is_finished_training():
            test_logger.start_epoch()
                with torch.set_grad_enabled(True):
                if self.use_intermediate_logits:
                    logits = model.forward(z).float()
                    probs = F.sigmoid(logits)
                else:
                    probs = model.forward(z).float()

                loss = loss_fn(probs, img_test).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # TODO: some convergence criteria/saver saving when model is done fine-tuning z_test
                # get out of loop and save best MSE which is just the loss rn

        test_logger.print_best_mse
