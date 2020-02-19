#! /usr/bin/env python

import datetime

import subprocess
import json

from argparse import ArgumentParser


def run(parsed):
    idx = parsed.index if parsed.index is None else int(parsed.index)

    with open(parsed.instructions,'r') as f:
        data = json.load(f)

    assert isinstance(data, list), "Instructions root must be a list"

    def is_comment(dct):
        return "comment" in dct.keys() and len(dct) == 1

    data = [dd for dd in data if not is_comment(dd)]

    def sanitize(string):
        string = string.replace('\"', '\\\"')
        string = string.replace('\'', '\\\'')
        return string

    def format_arg(k, arg):
        if k is None:
            kk = []
        else:
            kk = [k]

        if isinstance(arg, list):
            arg = [aa for a in arg for aa in format_arg(None, a)]
        if isinstance(arg, dict):
            arg = ["\"{}\"".format(sanitize(json.dumps(arg)))]
        #if isinstance(arg, str):
        #    arg = ["\"{}\"".format(arg)]

        if not isinstance(arg, list):
            arg = [arg]

        return kk + arg

    start = parsed.start
    end = parsed.end
    if end == -1:
        end = len(data)

    start_time = datetime.datetime.now()

    print("[BATCH ] Started at {}".format(start_time))

    for ii, d in enumerate(data):
        has = tuple(key in d for key in ("script", "program",))
        assert int(has[0]) + int(has[1]) == 1, "Instruction must have either 'script' or 'program' but not both"

        if "script" in d:
            call = d["script"]

        if "program" in d:
            call = d["program"]

        if not isinstance(call, list):
            call = [call]

        run = ii >= start and ii < end and (idx is None or ii == idx)
        if run:
            print("[BATCH ]", "Running instruction {}/{} ({}).".format(ii+1, len(data), " ".join(str(c) for c in call)))
            args = d.get("args", [])
            kwargs = d.get("kwargs", {})

            args = [str(e) for v in args for e in format_arg(None, v)]
            kwargs = [str(e) for k, v in kwargs.items() for e in format_arg(k, v)]

            ret_val = subprocess.call(call + args + kwargs)
            if ret_val != 0:
                raise ValueError("Instruction exited with errors. (err_code: {})".format(ret_val))
                exit(ret_val)
        else:
            print("[BATCH ]", "Skipped instruction {}/{} ({}).".format(ii+1, len(data), " ".join(str(c) for c in call)))

    print("[BATCH ]", "Done running {}.".format(parsed.instructions))

    end_time = datetime.datetime.now()
    print("[BATCH ] Done at {}".format(end_time))
    print("[BATCH ] Took {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = ArgumentParser(description='Load JSON instructions.')

    parser.add_argument("instructions", help= "Name of the INSTRUCTIONS json file to use.",  metavar="INSTRUCTIONS")

    parser.add_argument("--index", default=None, help= "Execute instruction INDEX idx.",  metavar="INDEX")
    parser.add_argument("--start", type=int, default=0, help= "Execute instruction from START index.",  metavar="START")
    parser.add_argument("--end", type=int, default=-1, help= "Execute instruction up to END (exclusive).",  metavar="END")

    parsed = parser.parse_args()
    run(parsed)
    exit()
