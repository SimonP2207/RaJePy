# -*- coding: utf-8 -*-
"""
RAdio JEts in PYthon test program.
"""
import os
import sys
import argparse
import shutil
import time
from RaJePy import JetModel, Pipeline, logger
from RaJePy import cfg

if __name__ == '__main__':
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("model_param_file",
                            help="Full path to model parameter file",
                            type=str)
        parser.add_argument("pipeline_param_file",
                            help="Full path to pipeline parameter file",
                            type=str)
        parser.add_argument("-v", "--verbose",
                            help="Increase output verbosity",
                            action="store_true")
        parser.add_argument("-rt", "--radiative-transfer",
                            help="Compute radiative transfer solutions",
                            action="store_true")
        parser.add_argument("-so", "--simobserve",
                            help="Conduct synthetic observations using CASA",
                            action="store_true")
        parser.add_argument("-r", "--resume",
                            help="Resume previous pipeline run if present",
                            action="store_true")
        parser.add_argument("-c", "--clobber",
                            help="Overwrite any data products/files present",
                            action="store_true")

        args = parser.parse_args()
        jet_param_file = os.path.abspath(args.model_param_file)
        pline_param_file = os.path.abspath(args.pipeline_param_file)
        verbose = args.verbose

        # Load model directory
        if os.path.dirname(args.pipeline_param_file) not in sys.path:
            sys.path.append(os.path.dirname(args.pipeline_param_file))
        jp = __import__(os.path.basename(args.pipeline_param_file)[:-3])

        # Set up common log for JetModel and Pipeline instances
        log_name = "ModelRun_"
        log_name += time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_name += ".log"
        logfile = os.sep.join([jp.params['dcys']['model_dcy'], log_name])
        log = logger.Log(fname=logfile)

        # Make sure model directory exists as logger requires it
        if not os.path.exists(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))

        pline = Pipeline(JetModel(jet_param_file, log=log), pline_param_file,
                         log=log)
        pline.log.add_entry("INFO", "Pipeline initiated using model parameters "
                            "defined in {}, and pipeline parameters defined in"
                            " {}".format(os.path.abspath(jet_param_file),
                                         os.path.abspath(pline_param_file)))
        pline.execute(resume=args.resume, clobber=args.clobber,
                      simobserve=args.simobserve, verbose=args.verbose,
                      dryrun=not args.radiative_transfer)
    else:
        # Testing files if needed
        jet_param_file = os.sep.join([cfg.dcys['files'],
                                      'example-model-params.py'])
        pline_param_file = os.sep.join([cfg.dcys['files'],
                                        'example-pipeline-params.py'])

        pline = Pipeline(JetModel(jet_param_file), pline_param_file)
        pline.execute(resume=False, clobber=False)


    for f in (jet_param_file, pline_param_file):
        dest = os.sep.join([pline.params['dcys']['model_dcy'],
                            os.path.basename(f)])
        dest = os.path.expanduser(dest)
        if f != dest:
            shutil.copyfile(f, dest)
