#ifdef USE_VK_IMGUI

#define VK_NO_PROTOTYPES

#include "imgui.h"
#include "backends/imgui_impl_vulkan.cpp"
#include "tr_local.h"

static VkDescriptorPool imgui_descriptor_pool;


void vk_imgui_initialize( void ) {
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000 }
    };

    VkDescriptorPoolCreateInfo info = {};
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets       = 1000 * ARRAY_LEN( pool_sizes );
    info.poolSizeCount = ARRAY_LEN( pool_sizes );
    info.pPoolSizes    = pool_sizes;

    VK_CHECK( qvkCreateDescriptorPool( vk.device, &info, NULL, &imgui_descriptor_pool ) );

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // High-DPI / 4K scaling: ImGui's defaults (13px font, tight padding) are authored
    // for ~1080p and look microscopic at a 3840-wide target. Derive a scale from the
    // window width and apply it to both the font size and all style metrics.
    // ScaleAllSizes must be called exactly once (it multiplies the style in place), so
    // it belongs here at init -- never in vk_imgui_draw, where it would compound every
    // frame. gls.windowWidth is already set by vk_create_window before this runs.
    {
        ImGuiIO& io = ImGui::GetIO();
        float uiScale = (float)gls.windowWidth / 1920.0f;   // ~2.0 at 3840 wide
        if ( uiScale < 1.0f )
            uiScale = 1.0f;

        ImFontConfig fontCfg;
        fontCfg.SizePixels = 13.0f * uiScale;
        io.Fonts->AddFontDefault( &fontCfg );               // baked into the atlas below

        ImGui::GetStyle().ScaleAllSizes( uiScale );
    }

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance       = vk.instance;
    init_info.PhysicalDevice = vk.physical_device;
    init_info.Device         = vk.device;
    init_info.QueueFamily    = vk.queue_family_index;
    init_info.Queue          = vk.queue;
    init_info.DescriptorPool = imgui_descriptor_pool;
    // r_fbo 0 renders straight to the swapchain via render_pass.main and returns early
    // from vk_create_render_passes (gamma + post passes never created); r_fbo 1 renders
    // offscreen and blits to the swapchain via render_pass.gamma. Pick whichever is the
    // actual final swapchain pass so ImGui's pipeline is valid + present-format compatible.
    // (Both are 1-sample unless MSAA is enabled in r_fbo 0 mode, in which case MSAASamples
    // below would also need to match the main pass -- revisit if we hit that.)
    init_info.RenderPass     = ( vk.render_pass.gamma != VK_NULL_HANDLE )
                                   ? vk.render_pass.gamma
                                   : vk.render_pass.main;
    init_info.MinImageCount  = 2;
    init_info.ImageCount     = vk.swapchain_image_count;
    init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;

    // The dynamic-loader bridge — MUST precede Init. VK_NO_PROTOTYPES means the
    // backend has no Vulkan symbols; feed it the renderer's already-loaded ones.
    ImGui_ImplVulkan_LoadFunctions(
        []( const char *name, void *user_data ) -> PFN_vkVoidFunction {
            PFN_vkVoidFunction p = qvkGetDeviceProcAddr( vk.device, name );
            return p ? p : qvkGetInstanceProcAddr( vk.instance, name );
        },
        NULL );

    ImGui_ImplVulkan_Init( &init_info );
    ImGui_ImplVulkan_CreateFontsTexture();

    Com_Printf( "Initialized ImGui with Vulkan backend\n" );
}

void vk_imgui_begin_frame( void ) {

}

void vk_imgui_draw( void ) {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2( (float)gls.windowWidth, (float)gls.windowHeight );
    io.DeltaTime = 1.0f / 60.0f;

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos ( ImVec2( 0.0f, 0.0f ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( ImVec2( io.DisplaySize.x / 3.0f, io.DisplaySize.y ), ImGuiCond_Always );

    ImGui::Begin( "RT Debug", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse );
    ImGui::Text( "ImGui overlay online (%.0f x %.0f)", io.DisplaySize.x, io.DisplaySize.y );
    ImGui::End();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData( ImGui::GetDrawData(), vk.cmd->command_buffer );

}

void vk_imgui_shutdown( void ) {
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
    qvkDestroyDescriptorPool( vk.device, imgui_descriptor_pool, NULL );
}

#endif