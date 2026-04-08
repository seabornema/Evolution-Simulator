#include <header/camera.h>


Camera::Camera(float zoom, int width, int height, glm::vec3 position){
    Camera::width = width;
    Camera::height = height;
    Position = position;  
    Zoom = zoom;
}

void Camera::updateMatrix(float width,float height, float nearPlane, float farPlane){
    glm::mat4 projection = glm::ortho(-width*0.5f, width*0.5f, -height*0.5f, height*0.5f, nearPlane, farPlane);
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, Position);
    view = glm::scale(view, glm::vec3(Zoom, Zoom, 1.0f)); 
    cameraMatrix = projection*view;
 
}

void Camera::Matrix(Shader& shader,const char* uniform){
    glUniformMatrix4fv(glGetUniformLocation(shader.ID,uniform),1,GL_FALSE,glm::value_ptr(cameraMatrix));
}

void Camera::Inputs(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
        Position -= speed*Up;
    }
     if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
        Position += speed*glm::normalize(glm::cross(Orientation,Up));
    }
     if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
        Position += speed*Up;
    }
     if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
        Position -= speed*glm::normalize(glm::cross(Orientation,Up));
    }
     if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS){
        Zoom += speed;
    }

     if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS){
        Zoom -= speed;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
        speed = 4.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE){
        speed = 1.0f;
    }
}
