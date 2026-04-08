#version 330 core
layout (location = 0) in vec3 aPos; 
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 _center;
layout (location = 3) in float _radius;

out vec3 ourColor;
out vec2 fragPos;
out vec2 center;
out float radius;

uniform mat4 camMatrix;


void main()
{
	gl_Position = camMatrix*vec4(aPos, 1.0);
    ourColor = aColor;
    fragPos = aPos.xy;
    center = _center;
    radius = _radius;
}   
