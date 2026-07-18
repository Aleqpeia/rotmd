"""Post-extract analyses.

Everything here *consumes* the artifacts ``rotmd extract`` wrote (``.npz`` +
``meta.json``) and the window ``rotmd equilibrate`` decided (``window.json``).
Nothing here re-reads a trajectory except the two stages that physically
cannot avoid it (DSSP and the local electrostatics need atoms beyond CA).
"""
