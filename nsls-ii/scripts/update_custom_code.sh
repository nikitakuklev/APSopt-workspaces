#!/bin/bash
# Run this script anytime we need to update the code to latest version
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/pysdds
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/Xopt
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/pybeamtools
# this is a token to private repo, read-only, ok to be public
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://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@github.com/nikitakuklev/apsopt
#python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps -e git+https://github.com/nikitakuklev/pyiota
#python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps -e git+https://github.com/nikitakuklev/ocelot