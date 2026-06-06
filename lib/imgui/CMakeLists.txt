# Bundled Dear ImGui (v1.91.4), built as a static, position-independent library and
# linked into rd-vulkant for the debug overlay. Mirrors EternalJK's bundled_libimgui.
#
# NOTE: only the ImGui *core* is compiled here. The Vulkan + SDL2 backends
# (include/backends/imgui_impl_{vulkan,sdl2}.cpp) are NOT listed -- they are
# unity-#included into the renderer's vk_imgui_context.cpp, so they inherit that
# TU's VK_NO_PROTOTYPES define + bundled Vulkan/SDL2 headers (the renderer loads
# vulkan-1.dll dynamically). IMGUI_INCLUDE_DIRS points at include/ so both
# `#include "imgui.h"` and `#include <backends/imgui_impl_vulkan.cpp>` resolve.

include_directories(include/)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(MSVC)

add_library(bundled_libimgui STATIC	include/imconfig.h
									include/imgui.cpp
									include/imgui.h
									include/imgui_demo.cpp
									include/imgui_draw.cpp
									include/imgui_internal.h
									include/imgui_tables.cpp
									include/imgui_widgets.cpp
									include/imstb_rectpack.h
									include/imstb_textedit.h
									include/imstb_truetype.h
                                  )

# Export in the standard CMake way (shadow the cache vars, like FindPNG does), so
# the renderer can pull them via ${IMGUI_INCLUDE_DIRS} / ${IMGUI_LIBRARIES}.
set(IMGUI_LIBRARIES bundled_libimgui PARENT_SCOPE)
set(IMGUI_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/lib/imgui/include/ PARENT_SCOPE)

mark_as_advanced(IMGUI_LIBRARIES IMGUI_INCLUDE_DIRS)

# Linked into a shared library (rd-vulkant), so build as PIC.
set_property(TARGET bundled_libimgui PROPERTY POSITION_INDEPENDENT_CODE True)
