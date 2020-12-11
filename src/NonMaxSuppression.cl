__kernel void nonMaxSuppression(
	__global uchar* src, int src_step, int src_offset, int src_rows, int src_cols)
{
	int y = get_global_id(0);
	int x = get_global_id(1);
	const YMAX = src_rows;
	const XMAX = src_cols;
	int q = 0;
	int r = 0;
	int edge_x_y = src[(x) * src_step + (y) + src_offset];
	int edge_x_yadd1;
	int edge_xadd1_y;
	float tan;

	if (y < YMAX && x < XMAX){
	
		if (y + 1 >= YMAX) {
			edge_x_yadd1 = edge_x_y;
		}
		else {
			edge_x_yadd1 = src[(x) * src_step + (y+1) + src_offset];
		}

		if (x + 1 >= XMAX) {
			edge_xadd1_y = edge_x_y;
		}
		else {
			edge_xadd1_y = src[(x+1) * src_step + (y) + src_offset];
		}

		if (edge_xadd1_y - edge_x_y != 0) {
			tan = (float)(-(edge_x_yadd1 - edge_x_y)) / (float)(edge_xadd1_y - edge_x_y);
		}
		else {
			tan = 0.0;
		}

		if ((tan >= 0 && tan < 0.4142) || (tan >= -0.4142 && tan <= 0)) {
			if (x + 1 < XMAX) {
				q =src[(x+1) * src_step + (y) + src_offset];
			}
			if (x - 1 >= 0) {
				r = src[(x-1) * src_step + (y) + src_offset];
			}
		}
		else if ((tan >= 0.4142 && tan < 2.4142)) {
			if (y - 1 >= 0 && x + 1 < XMAX) {
				q = src[(x+1) * src_step + (y-1) + src_offset];
			}
			if (y + 1 < YMAX && x - 1 >= 0) {
				r = src[(x-1) * src_step + (y+1) + src_offset];
			}
		}
		else if ((tan >= 2.4142 || tan <= -2.4142)) {
			if (y - 1 >= 0) {
				q = src[(x) * src_step + (y-1) + src_offset];
			}
			if (y + 1 < YMAX) {
				r = src[(x) * src_step + (y+1) + src_offset];
			}
		}
		else if ((tan >= -2.4142 && tan < -0.4142)) {
			if (y - 1 >= 0 && x - 1 >= 0) {
				q = src[(x-1) * src_step + (y-1) + src_offset];
			}
			if (y + 1 < YMAX && x + 1 < XMAX) {
				r = src[(x+1) * src_step + (y+1) + src_offset];
			}
		}

		if (src[(x) * src_step + (y) + src_offset] < q ||
			src[(x) * src_step + (y) + src_offset] < r) {
			src[(x) * src_step + (y) + src_offset] = 0;
		}
	}
}