

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
            self.logger.log()
        self.logger.visualize()
        return

    def evaluate_z_test(self, curr_model, img_test=None, **kwargs):
        # TODO: incl param that checks whether to run this test - prolly every n steps
        if z_test is None:
            # Sample random z vector, same shape as input
            # OK if same as input b/c on held out test img
            z_test = util.create_input_vec_thing()

            # Will need to backprop into the input
            z_test.requires_grad_()

        if img_test is None:
            # Get test image(s)
            # Ensure not the same as any in training

        # Make deep copy of model, so we can freeze it
        model = curr_model.deepcopy()
        
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        model = nn.DataParallel(model, args.gpu_ids)
        model = model.to(args.device)

        # Use model in eval mode for dropout/BN layers to behave their best
        model.eval()
        
        # Get optimizer and loss
        parameters = model.parameters()
        optimizer = util.get_optimizer(parameters, args)
        loss_fn = util.get_loss_fn(args.loss_fn, args)

        #TODO: maybe look at ckpt loading code
        test_logger.init # TODO: need to some kind of loop/iters for this

        # Get logger and saver
        logger = TestLogger(args)
        #saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric)
        
        #print(f'Logs: {logger.log_dir}')
        #print(f'Ckpts: {args.save_dir}')

        # Train model

        # TODO: some convergence criteria/saver saving when model is done fine-tuning z_test
        # get out of loop and save best MSE which is just the loss rn
        logger.log_hparams(args)
        while not logger.is_finished_training():
            logger.start_epoch()

            with torch.set_grad_enabled(True):
                if args.use_intermediate_logits:
                    logits = model.forward(z_test).float()
                    probs = F.sigmoid(logits) 
                else:
                    probs = model.forward(z_test).float()

                loss = torch.zeros(1, requires_grad=True).to(args.device)
                loss = loss_fn(probs, img_test).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            metrics = {'z_test_loss': loss.item()}
            #saver.save(logger.epoch, model, optimizer, args.device, metric_val=metrics.get(args.best_ckpt_metric, None))         
            
            # TODO: Log and visualize z_test, probs, target; write loss
            logger.log_status(inputs=z_test,
                              targets=img_test,
                              probs=probs,
                              save_preds=args.save_preds,
                              )

            logger.end_epoch()
            
        # Last log after everything completes
        logger.log_status(inputs=z_test,
                          targets=img_test,
                          probs=probs,
                          save_preds=args.save_preds,
                          force_visualize=True,
                          )


