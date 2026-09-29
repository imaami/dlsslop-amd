# Writes the SPIR-V module SPIRV into HEADER as `const unsigned char SYMBOL[]`,
# the array shape DXC's -Fh writes and the layer includes.
file(READ "${SPIRV}" bytes HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
file(WRITE "${HEADER}" "const unsigned char ${SYMBOL}[] = {${bytes}};\n")
