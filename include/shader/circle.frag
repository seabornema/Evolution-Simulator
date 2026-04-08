#version 330 core
out vec4 FragColor;  

in vec3 ourColor;
in vec2 fragPos;
in vec2 center;
in float radius;

void main()
{

    vec2 d = fragPos - center;
    float dist2 = dot(d, d);
    if(dist2 > radius*radius) discard;
    
    FragColor = vec4(ourColor, 1.0);
}
