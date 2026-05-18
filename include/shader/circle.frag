#version 330 core
out vec4 FragColor;  
in vec3 ourColor;
in vec2 fragPos;
in vec2 center;
in float radius;
in float theta;

void main()
{
    vec2 d = fragPos - center;
    float dist2 = dot(d, d);
    
    if (dist2 > radius * radius) discard;

    vec2 facing = vec2(cos(theta), sin(theta));

    float along  = dot(d, facing);         
    float angle = acos(clamp(along / length(d), -1.0, 1.0));
 
    if (along > 0.0 && angle < 0.3) {
        FragColor = vec4(1.0 - ourColor, 1.0);
    } else {
        FragColor = vec4(ourColor, 1.0);
    }
}
