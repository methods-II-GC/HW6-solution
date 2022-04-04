#!/usr/bin/env python
"""Computes bits per character entropy for strings."""

import argparse
import csv
import math
import sys

import pynini


M_LN2 = math.log(2)


class Error(Exception):
    pass


def _parse_token_type(token_type: str) -> pynini.TokenType:
    """Parses token type string.

    Args:
      token_type: token type, or the path to a symbol table.

    Returns:
      A token type string, or a symbol table.
    """
    if token_type in ["byte", "utf8"]:
        return token_type
    return pynini.SymbolTable.read_text(token_type)


def _bits_per_char(string: pynini.Fst, lm: pynini.Fst) -> float:
    """Computes bits per char according to LM FST.

    Args:
      string: an FSA to be scored.
      lm: a WFSA language model.

    Returns:
      The score as bits per char.

    Raises:
      Error: Composition failure.
    """
    # Checks properties ahead of time, just to be sure.
    eprops = pynini.ACCEPTOR | pynini.STRING | pynini.UNWEIGHTED
    oprops = string.properties(eprops, True)
    assert eprops == oprops, f"{oprops} != {eprops}"
    # Scores the lattice.
    lattice = pynini.intersect(string, lm)
    # Detects composition failure.
    if lattice.start() == pynini.NO_STATE_ID:
        raise Error("Composition failure")
    # The shortest backwards distance from the start state to a final state
    # is the cost of a string w.r.t. the LM.
    cost = pynini.shortestdistance(lattice, reverse=True)[lattice.start()]
    # Converts this to base-2.
    bits = float(cost) / M_LN2
    # A n-char string FSA has n + 1 states. Draw it if you don't believe me.
    chars = string.num_states() - 1
    return bits / chars


def main(args: argparse.Namespace) -> None:
    token_type = _parse_token_type(args.token_type)
    lm = pynini.Fst.read(args.lm)
    with open(args.corpus, "r") as source:
        tsv_writer = csv.writer(sys.stdout, delimiter="\t")
        for line in source:
            line = pynini.escape(line.rstrip())
            string = pynini.accep(line, token_type=token_type)
            try:
                bits_per_char = _bits_per_char(string, lm)
                tsv_writer.writerow([bits_per_char, line])
            except Error:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="input file path")
    parser.add_argument("--lm", required=True, help="input LM FST path")
    parser.add_argument(
        "--token_type",
        default="byte",
        help="token type, or path to symbol table (default: %(default)s)",
    )
    main(parser.parse_args())
