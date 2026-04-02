#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

struct PhysicsVertex {
    vec3 pos;
    float invMass;
    vec3 vel;
    float padding;
    vec3 pos_n;
    float padding2;
    vec3 vel_n;
    float padding3;
    vec4 normal;
};

struct RenderVertex {
    vec4 color;
};

layout(std430, binding = 1) readonly buffer PhysicsBuffer {
    PhysicsVertex physicsVerts[];
};

layout(std430, binding = 2) readonly buffer RenderBuffer {
    RenderVertex renderVerts[];
};
layout(location = 0) out vec3 fragColor;

void main() {
    gl_PointSize = 3.0;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(physicsVerts[gl_VertexIndex].pos, 1.0);
    fragColor = renderVerts[gl_VertexIndex].color.xyz;
}
