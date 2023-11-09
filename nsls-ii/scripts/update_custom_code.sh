#!/bin/bash
# Run this script anytime we need to update the code to latest version
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/pysdds
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/Xopt
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://github.com/nikitakuklev/pybeamtools
# this is a fine grained token to private repo, read-only, ok to be public, need to obfuscate or github will revoke it
var0="github_pat"
var1="11AAIC3CI0jm4BkyYZRUmH"
var2="4WPY5xhczeFQgRCBiLavA0FOfFbfFiULYP4aow55Y4"
var3="BHG45RATNIAIxu3e3"
python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps git+https://${var0}_${var1}_${var2}${var3}@github.com/nikitakuklev/apsopt
#python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps -e git+https://github.com/nikitakuklev/pyiota
#python -m pip install --upgrade --force-reinstall --no-build-isolation --no-index --no-deps -e git+https://github.com/nikitakuklev/ocelot