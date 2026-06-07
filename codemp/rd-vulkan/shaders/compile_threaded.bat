@echo off

set "VSCMD_START_DIR=%CD%"
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

rem 'CL' is a reserved MSVC env var (cl.exe prepends it to its command line) - clear any inherited value
set "CL="

set tools_dir=tools
set bh=%tools_dir%\compile_threaded.exe
set bh_cpp=%tools_dir%\compile_threaded.cpp

if exist %bh% (
    del /Q %bh%
)

if not exist %bh% (
    cl.exe /EHsc /nologo /Fe%tools_dir%\ /Fo%tools_dir%\ %bh_cpp%
)

if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
set glsl=glsl\\
set glslang=%VULKAN_SDK%\Bin\glslangValidator.exe
::set glslang=%tools_dir%\glslang\bin\glslangValidator.exe
set outf=+spirv\shader_data.c
set outfb=+spirv\shader_binding.c

"%bh%" "%glsl%" "%glslang%" "%outf%" "%outfb%"
