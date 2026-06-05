#pragma once

namespace xai {

struct Model;

void build_rope(Model *m);
void apply_rope(float *vec, int hd, int pos,
                const float *cos_tbl, const float *sin_tbl);

} // namespace xai