#!/bin/sh
# Cargo runner: ad-hoc signs the binary with the Hypervisor.framework entitlement, then runs it.
set -e
codesign -s - -f --entitlements "$(dirname "$0")/hypervisor.entitlements" "$1" 2>/dev/null
exec "$@"
