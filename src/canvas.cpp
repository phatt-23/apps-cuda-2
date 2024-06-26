#include "inc/canvas.h"
#include "inc/module.h"

Canvas::Canvas(const char* name, cv::Mat& cv_background, uint2 canvas_size, size_t flags)
    : name(name)
    , cv_canvas(cv::Mat::zeros(cv::Size(canvas_size.y, canvas_size.x), flags)) 
    , size({canvas_size.y, canvas_size.x}) 
    , bg_image(cv_background)
    , canvas(cv_canvas)  
{
    this->flush();
    std::cout << "INFO: Canvas created! (size = {" 
        << this->size.y << "," << this->size.x << "}, type = " 
        << ((flags == CV_8UC3) ? "RGB" : "RGBA") << ")" << std::endl;
}

Canvas::~Canvas()
{
    std::cout << "INFO: Canvas freed!" << std::endl;
}

void Canvas::draw(CudaRect& rect) // draw a CudaRect to the canvas CudaImg
{
    uint2 pos = { .x= (uint32_t)rect.pos.x, .y= (uint32_t)rect.pos.y };

    if(rect.rsize.x == 0) {
        cu_insert_rgba_image(this->canvas, rect, pos, rect.alpha);
        return;
    }

    CudaImg og(*rect.p_cv_mat);
    cv::Mat cv_resized = cv::Mat(cv::Size(rect.rsize.x, rect.rsize.y), (rect.channels - 1)*8);
    CudaImg resized(cv_resized);

    cu_bilinear_resize(resized, og);
    cu_insert_rgba_image(this->canvas, resized, pos, rect.alpha);
}

void Canvas::flush() // inserts the default background CudaImg
{
    cu_bilinear_resize(this->canvas, this->bg_image);
}

void Canvas::show()
{
    cv::imshow(this->name, this->cv_canvas);
}