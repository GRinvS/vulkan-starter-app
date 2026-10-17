#include "application.hpp"

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

    namespace {

        // ќтключаем culling Ч с ним были проблемы с намоткой.
        // ƒл€ одного куба это визуально ничем не хуже.
        constexpr bool DEBUG_NO_CULLING = false;

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 color;

            static VkVertexInputBindingDescription bindingDescription() {
                return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            }

            static std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions() {
                return { {
                    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                    { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                } };
            }
        };

        struct UniformBufferObject {
            glm::mat4 model;
            glm::mat4 view;
            glm::mat4 proj;
        };

        struct ColorUniform {
            glm::vec4 userColor;
        };

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_vertex_buffer = VK_NULL_HANDLE;
        uint32_t vertex_count = 0;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_index_buffer = VK_NULL_HANDLE;
        uint32_t index_count = 0;

        VkBuffer vk_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_uniform_buffer = VK_NULL_HANDLE;
        void* uniform_buffer_mapped = nullptr;

        VkBuffer vk_color_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_color_uniform_buffer = VK_NULL_HANDLE;
        void* color_uniform_buffer_mapped = nullptr;

        // “рансформации куба
        glm::vec3 obj_position(0.0f);
        glm::vec3 obj_rotation(0.0f);
        glm::vec3 obj_scale(1.0f);

        // ‘иксированные параметры камеры/проекции Ч без UI
        constexpr float camera_distance = 6.0f;

        // ---------- ”тилиты ----------
        std::vector<char> readFile(const std::string& path) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << path << '\n';
                return {};
            }
            size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = code.size();
            info.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module\n";
                return VK_NULL_HANDLE;
            }
            return module;
        }

        std::vector<Vertex> generateCube(float size) {
            const float h = size * 0.5f;
            std::vector<Vertex> vertices = {
                { { -h, -h,  h }, { 1.0f, 0.2f, 0.2f } },  // красный
                { {  h, -h,  h }, { 0.2f, 1.0f, 0.2f } },  // зелЄный
                { {  h,  h,  h }, { 0.2f, 0.2f, 1.0f } },  // синий
                { { -h,  h,  h }, { 1.0f, 1.0f, 0.2f } },  // жЄлтый
                { { -h, -h, -h }, { 1.0f, 0.2f, 1.0f } },  // пурпурный
                { {  h, -h, -h }, { 0.2f, 1.0f, 1.0f } },  // голубой
                { {  h,  h, -h }, { 1.0f, 1.0f, 1.0f } },  // белый
                { { -h,  h, -h }, { 0.9f, 0.5f, 0.1f } },  // оранжевый
            };
            return vertices;
        }

        std::vector<uint32_t> generateCubeIndices() {
            return {
                0, 1, 2,  2, 3, 0,   // front
                1, 5, 6,  6, 2, 1,   // right
                5, 4, 7,  7, 6, 5,   // back
                4, 0, 3,  3, 7, 4,   // left
                3, 2, 6,  6, 7, 3,   // top
                4, 5, 1,  1, 0, 4,   // bottom
            };
        }

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr) {
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            if (mapped) {
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
            }

            VmaAllocationInfo info{};
            if (vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo,
                &allocInfo, &buffer, &allocation, &info) != VK_SUCCESS) {
                std::cerr << "Failed to create buffer\n";
                return false;
            }

            if (mapped) {
                *mapped = info.pMappedData;
                if (*mapped == nullptr) {
                    std::cerr << "WARNING: buffer mapped pointer is nullptr\n";
                }
            }
            return true;
        }

        bool createGraphicsPipeline() {
            VkDevice device = graphics::internal::context.device;

            auto vertCode = readFile("shaders/cube.vert.spv");
            auto fragCode = readFile("shaders/cube.frag.spv");
            if (vertCode.empty() || fragCode.empty()) {
                std::cerr << "Shader SPIR-V not found. Compile shaders first with glslc.\n";
                return false;
            }

            VkShaderModule vertModule = createShaderModule(vertCode);
            VkShaderModule fragModule = createShaderModule(fragCode);
            if (!vertModule || !fragModule) return false;

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName = "main";

            auto binding = Vertex::bindingDescription();
            auto attrs = Vertex::attributeDescriptions();
            VkPipelineVertexInputStateCreateInfo vertexInput{};
            vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount = 1;
            vertexInput.pVertexBindingDescriptions = &binding;
            vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vertexInput.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_BACK_BIT;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlend{};
            colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlend.attachmentCount = 1;
            colorBlend.pAttachments = &colorBlendAttachment;

            VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = dynamicStates;

            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 2;
            layoutInfo.pBindings = bindings;
            if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vk_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor set layout\n";
                return false;
            }

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &vk_descriptor_set_layout;
            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState = &multisample;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlend;
            pipelineInfo.pDynamicState = &dynamic;
            pipelineInfo.layout = vk_pipeline_layout;
            pipelineInfo.renderPass = graphics::internal::context.render_pass;
            pipelineInfo.subpass = 0;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            vkDestroyShaderModule(device, vertModule, nullptr);
            vkDestroyShaderModule(device, fragModule, nullptr);
            return true;
        }

        bool createDescriptorSets() {
            VkDevice device = graphics::internal::context.device;

            VkDescriptorPoolSize poolSizes[1]{};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[0].descriptorCount = 2;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = poolSizes;
            poolInfo.maxSets = 1;
            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = vk_descriptor_pool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &vk_descriptor_set_layout;
            if (vkAllocateDescriptorSets(device, &allocInfo, &vk_descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate descriptor set\n";
                return false;
            }

            VkDescriptorBufferInfo uboInfo{ vk_uniform_buffer, 0, sizeof(UniformBufferObject) };
            VkDescriptorBufferInfo colorInfo{ vk_color_uniform_buffer, 0, sizeof(ColorUniform) };

            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = vk_descriptor_set;
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &uboInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = vk_descriptor_set;
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[1].descriptorCount = 1;
            writes[1].pBufferInfo = &colorInfo;

            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
            return true;
        }

        void updateUniformBuffer(const glm::vec3& position,
            const glm::vec3& rotation, const glm::vec3& scale) {
            auto& ctx = graphics::internal::context;

            glm::mat4 model(1.0f);
            model = glm::translate(model, position);
            model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1, 0, 0));
            model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0, 1, 0));
            model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0, 0, 1));
            model = glm::scale(model, scale);

            float aspect = 1.0f;
            if (ctx.swapchain_extent.height > 0) {
                aspect = static_cast<float>(ctx.swapchain_extent.width) /
                    static_cast<float>(ctx.swapchain_extent.height);
            }

            glm::mat4 view = glm::lookAt(glm::vec3(0, 0, camera_distance),
                glm::vec3(0, 0, 0),
                glm::vec3(0, 1, 0));

            // ѕроекци€ фиксированна€ Ч perspective, 45∞, near=0.1, far=100
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
            proj[1][1] *= -1.0f; // Vulkan Y-flip

            UniformBufferObject ubo{ model, view, proj };
            if (uniform_buffer_mapped) {
                std::memcpy(uniform_buffer_mapped, &ubo, sizeof(ubo));
            }

            // ÷вет всегда белый Ч фактический цвет берЄтс€ из vertex color
            ColorUniform cu{ glm::vec4(1.0f) };
            if (color_uniform_buffer_mapped) {
                std::memcpy(color_uniform_buffer_mapped, &cu, sizeof(cu));
            }
        }

    } // namespace

    bool initialize() {
        auto vertices = generateCube(2.0f);
        auto indices = generateCubeIndices();
        vertex_count = static_cast<uint32_t>(vertices.size());
        index_count = static_cast<uint32_t>(indices.size());

        void* vertexData = nullptr;
        if (!createBuffer(sizeof(Vertex) * vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk_vertex_buffer, vma_vertex_buffer, &vertexData)) {
            return false;
        }
        std::memcpy(vertexData, vertices.data(), sizeof(Vertex) * vertex_count);

        void* indexData = nullptr;
        if (!createBuffer(sizeof(uint32_t) * index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            vk_index_buffer, vma_index_buffer, &indexData)) {
            return false;
        }
        std::memcpy(indexData, indices.data(), sizeof(uint32_t) * index_count);

        if (!createBuffer(sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_uniform_buffer, vma_uniform_buffer, &uniform_buffer_mapped)) {
            return false;
        }
        if (!createBuffer(sizeof(ColorUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_color_uniform_buffer, vma_color_uniform_buffer,
            &color_uniform_buffer_mapped)) {
            return false;
        }

        if (!createGraphicsPipeline()) return false;
        if (!createDescriptorSets()) return false;

        updateUniformBuffer(obj_position, obj_rotation, obj_scale);

        std::cout << "Cube initialized OK\n";
        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);

        if (vk_descriptor_pool) vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
        if (vk_descriptor_set_layout) vkDestroyDescriptorSetLayout(ctx.device, vk_descriptor_set_layout, nullptr);
        if (vk_pipeline) vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
        if (vk_pipeline_layout) vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);

        if (vk_vertex_buffer) vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vma_vertex_buffer);
        if (vk_index_buffer) vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vma_index_buffer);
        if (vk_uniform_buffer) vmaDestroyBuffer(ctx.allocator, vk_uniform_buffer, vma_uniform_buffer);
        if (vk_color_uniform_buffer) vmaDestroyBuffer(ctx.allocator, vk_color_uniform_buffer, vma_color_uniform_buffer);

        vkQueueWaitIdle(ctx.graphics_queue);
    }

    void update(double /*time*/) {
        ImGui::Begin("Cube");

        ImGui::DragFloat3("Position", &obj_position.x, 0.05f);
        ImGui::DragFloat3("Rotation", &obj_rotation.x, 1.0f);
        ImGui::DragFloat3("Scale", &obj_scale.x, 0.05f, 0.1f, 5.0f);

        if (ImGui::Button("Reset")) {
            obj_position = glm::vec3(0.0f);
            obj_rotation = glm::vec3(0.0f);
            obj_scale = glm::vec3(1.0f);
        }

        ImGui::End();

        updateUniformBuffer(obj_position, obj_rotation, obj_scale);
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        VkCommandBuffer cmd = fd.command_buffer;
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkClearValue clearValues[2]{};
        clearValues[0].color = { { 0.05f, 0.05f, 0.08f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass = ctx.render_pass;
        rpBegin.framebuffer = fd.framebuffer;
        rpBegin.renderArea = { { 0, 0 }, ctx.swapchain_extent };
        rpBegin.clearValueCount = 2;
        rpBegin.pClearValues = clearValues;
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{ 0, 0,
            static_cast<float>(ctx.swapchain_extent.width),
            static_cast<float>(ctx.swapchain_extent.height),
            0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{ { 0, 0 }, ctx.swapchain_extent };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        VkBuffer vertexBuffers[] = { vk_vertex_buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
            0, 1, &vk_descriptor_set, 0, nullptr);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application