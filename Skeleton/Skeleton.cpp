//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Schneider Ákos
// Neptun : XYUXUA
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

//ha harom koordinatat nem egerrel, hanem begepelve akarnank megadni
//#define TEST

//a haromszog kitoltesi szinenek animacioja
//#define COLOR

//a pontossag amivel floatokat osszehasonlitunk
float floatEqualityPrecision = 0.00001f;

//igazat ad, ha a p1 es p2 altal krealt egyenesre raesik a p3
inline bool isPointOnLine(vec2 p1, vec2 p2, vec2 p3) 
{
	vec2 va = p1 - p2;
	vec2 vb = p3 - p2;
	float area = va.x * vb.y - va.y * vb.x;
	if (fabs(area) < floatEqualityPrecision)
		return true;
	return false;
}

inline float cross(const vec2& p1, const vec2& p2) {
	return p1.x * p2.y - p2.x * p1.y;
}

inline float orien(const std::vector<vec2>& polyg) {
	float sum = 0.0f;
	for (unsigned int i = 0; i < polyg.size(); ++i) {
		sum += cross(polyg[i], polyg[(i + 1) % polyg.size()]);
	}
	return sum;
}

struct Triangle {

	Triangle(const vec2& p1, const vec2& p2, const vec2& p3): 
		p1(p1), p2(p2), p3(p3) {}
	vec2 p1;
	vec2 p2;
	vec2 p3;

};

bool pointInTriangle(const vec2& p, const Triangle& tri) {

	return (orien({p, tri.p1, tri.p2}) > 0.0f) &&
		(orien({p, tri.p2, tri.p3}) > 0.0f) &&
		(orien({p, tri.p3, tri.p1}) > 0.0f);
}

//en.cppreference.com/w/cpp/algorithm/reverse
template <class BidirectionalIterator>
void MyReverse(BidirectionalIterator first, BidirectionalIterator last)
{
	while ((first != last) && (first != --last)) {
		std::iter_swap(first, last);
		++first;
	}
}

//en.cppreference.com/w/cpp/algorithm/find
template<class InputIt, class UnaryPredicate>
constexpr InputIt MyFind_if(InputIt first, InputIt last, UnaryPredicate p)
{
	for (; first != last; ++first) {
		if (p(*first)) {
			return first;
		}
	}
	return last;
}

std::vector<vec2> EarClipping(std::vector<vec2> polyg) {

	const bool isClockwise = (orien(polyg) > 0);

	if (!isClockwise) {
		MyReverse(std::begin(polyg),std::end(polyg));
	}

	std::vector<vec2> res;

	while (polyg.size() > 3) {

		const unsigned int size = polyg.size();

		bool isTriangRemoved = false;

		for (unsigned int i = 0; i < size; ++i) {
			const vec2 p1 = polyg[i];
			const vec2 p2 = polyg[(i + 1) % size];
			const vec2 p3 = polyg[(i + 2) % size];
			const bool isClockwise = (orien({ p1,p2,p3 }) > 0.0f);

			if (!isClockwise)
				continue;
			const bool hasPoint = (MyFind_if(std::begin(polyg), std::end(polyg),
				[&p1, &p2, &p3](const vec2& p)
				{
					return pointInTriangle(p, { p1,p2,p3 });
				}) != std::end(polyg));

			if (hasPoint) 
				continue;

			isTriangRemoved = true;
			res.push_back(p1);
			res.push_back(p2);
			res.push_back(p3);
			polyg.erase(std::begin(polyg) + (i + 1) % size);
		}
		if (!isTriangRemoved)
			break;
	}
	return res;
}

//floatok osszehasonlitasa
inline bool equals(float f1, float f2) {
	return fabs(float(f1 - f2)) < floatEqualityPrecision;
}

//vec2-k osszehasonlitasa
inline bool equalsVec2(const vec2& v1, const vec2& v2) {
	return equals(v1.x, v2.x) && equals(v1.y, v2.y);
}

//ket pont alapjan kiszamolja a helyes kor kozeppontjat
vec2 CountOtherCircleCenter(const vec2& p1, const vec2& p2) {
	//ketismeretlenes linearis egyenletrendszer megoldasa
	//a*cx+b*cy=e
	//c*cx+d*cy=f
	
	//equation nr.1  |n| * (r - ro) = 0
	float a = p1.x - p2.x;
	float b = p1.y - p2.y;
	float e = (p1.x * p1.x + p1.y * p1.y - p2.x * p2.x - p2.y * p2.y) / 2;

	//equation nr.2  |c|^2 - 1 = (p1-c)^2
	float c = 2 * p1.x;
	float d = 2 * p1.y;
	float f = p1.x * p1.x + p1.y * p1.y + 1;

	float determinant = a * d - b * c;
	//ha a determinans nulla, az azt jelenti, hogy akkor vagy ket pontot felvettunk egy helyre, vagy a ket pont altal 
	//meghatarozott egyenesre illeszkedik az origo
	if (equals(determinant, 0.0f)) {
			//error
			return vec2(-1000.0f, -1000.0f);
	}
	else {
		float cx = (e * d - b * f) / determinant;
		float cy = (a * f - e * c) / determinant;
		return vec2(cx, cy);
	}
	
}

class MyGPUProgram : public GPUProgram {

	// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
	const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

	// fragment shader in GLSL
	const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

public:
	MyGPUProgram() {

		// create program for the GPU
		create(vertexSource, fragmentSource, "outColor");
	}
};

MyGPUProgram * gpuProgram; // vertex and fragment shaders

//a szurke egysegkor osztalya
class Circle {
	unsigned int vao, vbo;    // vertex array object id
	float vertices[202];
public:
	Circle() {
		//the circle
		const float twoPi = 2.0f * float(M_PI);
		float delta = twoPi / 100;		//100 haromszogbol
		int j = 0;
		for (unsigned int i = 0; i <= 100; i++) {
				vertices[j++] = cosf(i * delta);
				vertices[j++] = sinf(i * delta);
		}
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vertices),  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		//szurke szin(mindharom ertek ugyanaz)
		int location = glGetUniformLocation(gpuProgram->getId(), "color");
		glUniform3f(location, 0.5f, 0.5f, 0.5f); // 3 floats

		
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 101 /*# Elements*/);
	}
};

//a kijelolt pontok megjelenitett pottyeinek osztalya
class Point {
	unsigned int vao, vbo;    // vertex array object id
	float vertices[22];
	vec2 cp;

public:
	Point(vec2 c): cp(c){}

	void setVertices(vec2 c, float r) {

		const float twoPi = 2.0f * float(M_PI);
		float delta = twoPi / 10;		
		int j = 0;
		for (unsigned int i = 0; i < 11; i++) {
			vertices[j++] = c.x + r * cosf(i * delta);
			vertices[j++] = c.y + r * sinf(i * delta);
		}
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
	}

	void Draw() {

		glBindVertexArray(vao);		// make it active

		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vertices),  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		// feher szinu
		int location = glGetUniformLocation(gpuProgram->getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats
		
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 11 /*# Elements*/);
	}
};

//sziriuszi oldal, harom ilyen oldalbol all a haromszog
class SiriusLine {
	float vertices[302];
	int size;
	vec2 cp;   //centerpoint
	float radius; //sugar

	//oldalhossz
	float linelength;

	//ket vegpontja a sziriuszi szakasznak
	std::vector<vec2> endpoints;

public:
	SiriusLine(){
		size = 302;
		linelength = 0.0f;
	}

	float getLineLength() {
		return linelength;
	}

	vec2 getCenterpoint() {
		return cp;
	}

	std::vector<vec2> getEndpoints() {
		return endpoints;
	}

	//oldalhossz szamitasa
	void countLineLength(const std::vector<vec2>& side) {
		float sum = 0.0f;
		for (unsigned int i = 0; i < side.size() - 1; i++) {
			vec2 dp(side[i + 1].x - side[i].x, side[i + 1].y - side[i].y);

			sum += sqrtf(dp.x * dp.x + dp.y * dp.y) / (1.0f - side[i].x * side[i].x - side[i].y * side[i].y);

		}
		linelength = sum;
	}

	std::vector<vec2> setVertices(const vec2& c, const vec2& p1, const vec2& p2, int id) {
		cp = c;
		endpoints.clear();
		endpoints.push_back(p1);
		endpoints.push_back(p2);

		//the radius of the imaginary circle
		float r = length(c - p1);
		radius = r;

		//a ket fazis, de ezekrol meg nem tudjuk melyiktol melyikig kell elmennunk
		float arc1 = atan2f((p1.y - c.y), (p1.x - c.x));
		float arc2 = atan2f((p2.y - c.y), (p2.x - c.x));

		//fontos a jo sorrend a kesobbi line_loop es fulvagas miatt
		bool goodOrder = true;
		if (arc1 > arc2) {
			goodOrder = false;
		}

		//a korivet hany szegmensre bontom
		int segments = (size-2)/2;

		//a kirajzolasnal levo fazisoknal ki kell valasztani melyik a kezdo es a vegso
		float arc_start = arc1 < arc2 ? arc1 : arc2;
		float arc_end = arc1 > arc2 ? arc1 : arc2;
		float arc_length = arc_end - arc_start;

		//ezzel megakadalyozzuk, hogy a megfelelo koriv pont masik oldalat rajzolja
		if (arc_length > M_PI) {
			arc_length = 2.0f * M_PI - arc_length;
			arc_start = arc_end;
			
			goodOrder = !goodOrder;
		}
		//egy szegmens merete
		float sizeOfSegment = arc_length / segments;

		//csucspontok a koriven
		int j = 0;
		for (int i = 0; i < size/2; i++) {
				vertices[j++] = cp.x + (r * cosf(i *(sizeOfSegment) + arc_start));
				vertices[j++] = cp.y + (r * sinf(i *(sizeOfSegment) + arc_start));
		}

		std::vector<vec2> result;

		//csucspontok jo sorrendben valo berakasa std::vectorba
		for (int i = 0; i < size; i += 2) {
			vec2 v(vertices[i], vertices[i + 1]);
			if (!goodOrder)
				result.insert(result.begin(), v);
			else
				result.push_back(v);
		}

		countLineLength(result);

		//ne legyen a sokszogben ketszer ugyanaz a pont
		result.pop_back();

		return result;
	}

};

//a kitoltes szinskalai
float r;
float g;
float b;

#ifdef COLOR
//szinvillogas
float incr_r = 0.03f;
float incr_g = 0.04f;
float incr_b = 0.05f;
#endif

//the green inside of the triangle
class SiriusTriangle {
	unsigned int vao, vbo;    // vertex array object id

	std::vector<vec2> endpoints;
	std::vector<vec2> fill;
	std::vector<SiriusLine*> sides;
	std::vector<vec2> outline;
	
	
public:
	SiriusTriangle() {}

	void addSide(SiriusLine* side) {
		sides.push_back(side);
	}

	void addEndPoint(const vec2& endpoint) {
		endpoints.push_back(endpoint);
	}

	void setFillingTriangles(std::vector<vec2> vtcs) {
		
		fill = EarClipping(vtcs);
		outline = vtcs;
	}

	void setToNull() {
		fill.erase(std::begin(fill), std::end(fill));
		outline.erase(std::begin(outline), std::end(outline));
	}


	void Create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		r = 0.00f;
		g = 0.33f;
		b = 0.66f;
	}

	//a kitoltes szinezese
	void Draw() {
		glBindVertexArray(vao);		// make it active

		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * fill.size(),  // # bytes
			&fill[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

#ifdef COLOR
		if (r > 1.0f) {
			incr_r = -0.03f;
		}
		else if (r < 0.0f) {
			incr_r = 0.03f;
		}

		if (g > 1.0f) {
			incr_g = -0.04f;
		}
		else if (g < 0.0f) {
			incr_g = 0.04f;
		}

		if (b > 1.0f) {
			incr_b = -0.05f;
		}
		else if (b < 0.0f) {
			incr_b = 0.05f;
		}
		r += incr_r;
		g += incr_g;
		b += incr_b;
#endif

#ifndef COLOR
		//turkisz kek
		r = 0.00f;
		g = 0.81f;
		b = 0.82f;
#endif

		int location = glGetUniformLocation(gpuProgram->getId(), "color");
		glUniform3f(location, r, g, b); // 3 floats

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, fill.size() /*# Elements*/);
	}

	//korvonal szinezese
	void DrawOutline() {
		glBindVertexArray(vao);		// make it active

		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * outline.size(),  // # bytes
			&outline[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		// Set color to (0.0, 1.0, 0.0) = green
		int location = glGetUniformLocation(gpuProgram->getId(), "color");
		glUniform3f(location, 1.0f, 0.0f, 0.0f); // 3 floats

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_LOOP, 0 /*startIdx*/, outline.size() /*# Elements*/);
	}

};

//szogszamolo fuggveny
std::vector<float> countAngle(const std::vector<vec2>& points) {

	std::vector<float> angles;
	std::vector<vec2> triangle = points;

	// orajarassal megfelelo iranyban kell lennie
	const bool isClockwise = (orien(triangle) > 0);
	if (!isClockwise) {
		MyReverse(std::begin(triangle), std::end(triangle));
	}

	vec2 cc1 = CountOtherCircleCenter(triangle[0], triangle[1]);
	vec2 cc2 = CountOtherCircleCenter(triangle[1], triangle[2]);
	vec2 cc3 = CountOtherCircleCenter(triangle[2], triangle[0]);

	//a vektorok, amik kozotti szogekre van szukseg
	vec2 v11 = cc3 - triangle[0];
	vec2 v12 = cc1 - triangle[0];

	vec2 v21 = cc1 - triangle[1];
	vec2 v22 = cc2 - triangle[1];

	vec2 v31 = cc2 - triangle[2];
	vec2 v32 = cc3 - triangle[2];
	
	//alpha
	if (cross(vec3(v11),vec3(v12)).z > 0.0f)
		angles.push_back(180.0f - (180.0f / M_PI) * acosf(dot(v11, v12) / (length(v11) * length(v12))));
	else
		angles.push_back((180.0f / M_PI) * acosf(dot(v11, v12) / (length(v11) * length(v12))));

	//beta
	if (cross(vec3(v21), vec3(v22)).z > 0.0f)
		angles.push_back(180.0f - (180.0f / M_PI) * acosf(dot(v21, v22) / (length(v21) * length(v22))));
	else
		angles.push_back((180.0f / M_PI) * acosf(dot(v21, v22) / (length(v21) * length(v22))));

	//gamma
	if (cross(vec3(v31), vec3(v32)).z > 0.0f)
		angles.push_back(180.0f - (180.0f / M_PI) * acosf(dot(v31, v32) / (length(v31) * length(v32))));
	else
		angles.push_back((180.0f / M_PI) * acosf(dot(v31, v32) / (length(v31) * length(v32))));

	return angles;
}

//egysegkor
Circle* circle = new Circle();

//a harom pont amit kijelol a felhasznalo
Point* p1 = new Point(vec2(0.0f, 0.0f));
Point* p2 = new Point(vec2(0.0f, 0.0f));
Point* p3 = new Point(vec2(0.0f, 0.0f));

//a harom oldala a sziriuszi haromszognek
SiriusLine* vonal1 = new SiriusLine();
SiriusLine* vonal2 = new SiriusLine();
SiriusLine* vonal3 = new SiriusLine();

//maga a haromszog
SiriusTriangle* triangle = new SiriusTriangle();


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	//the circle
	
	circle->Create();

	p1->setVertices(vec2(0.0f, 0.0f), 0.0f);
	p1->Create();

	p2->setVertices(vec2(0.0f, 0.0f), 0.0f);
	p2->Create();

	p3->setVertices(vec2(0.0f, 0.0f), 0.0f);
	p3->Create();

	triangle->Create();
	
	gpuProgram = new MyGPUProgram();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram->getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	//egysegkor
	circle->Draw();

	//kitoltes
	triangle->Draw();

	//korvonal
	triangle->DrawOutline();

	//pontok
	p1->Draw();
	p2->Draw();
	p3->Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') ;        // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	//ha egysegkoron kivulre kattintanank
	vec2 point(cX, cY);
	if (length(point) > 1.0f) return;
}

//a kijelolt pontok taroloja
std::vector<vec2> points;

//a sziriuszi oldalak pontjai (a felbontas)
std::vector<vec2> line1;
std::vector<vec2> line2;
std::vector<vec2> line3;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	//ha az egysegkoron kivulre kattintanank
	vec2 point(cX, cY);
	if (length(point) > 1.0f)
		return;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON: break;
	case GLUT_MIDDLE_BUTTON: break;
	case GLUT_RIGHT_BUTTON: break;
	}

	//makro a megadott pontok tesztelesehez, szoval ha nem kijeloljuk, hanem begepeljuk a koordinatakat
#ifdef TEST
		// POINT1: 
		if(points.size() == 0)
			points.push_back(vec2(0.64f, 0.4f));
#endif
	if (state == GLUT_DOWN) {
		
#ifndef TEST
		points.push_back(point);
#endif // !TEST

		//a pontok altal meghatarozott korok kozeppontjai
		vec2 circleCenter1;
		vec2 circleCenter2;
		vec2 circleCenter3;

		switch (points.size()) {
		case 1:
#ifdef TEST
			//POINT2: 
			points.push_back(vec2(-0.1f, 0.5f));
#endif		
			//ki kell nullazni a masik ketto pontot, hogy csak egy jelenjen meg
			p2->setVertices(vec2(0.0f, 0.0f), 0.00f);
			p3->setVertices(vec2(0.0f, 0.0f), 0.00f);

			//haromszog ne jelenjen meg
			triangle->setToNull();

			//pont beallitasa az elso klikk helyere
			p1->setVertices(points.at(0), 0.007f);
			break;
		case 2:
#ifdef TEST
			//POINT3: 
			points.push_back(vec2(-0.3f, -0.4f));
#endif
			circleCenter1 = CountOtherCircleCenter(points.at(0), points.at(1));

			//ha a masodfoku egyenlet megoldasanal nulla a determinans
			if (!equalsVec2(circleCenter1, vec2(-1000.0f, -1000.0f))) {
				line1 = vonal1->setVertices(circleCenter1, points.at(0), points.at(1), 0);
				p2->setVertices(points.at(1), 0.007f);
			}
			else {
				//ha a ket pont altal meghatarozott egyenesre illeszkedik az origo, akkor csak egy egyenest rajzolunk a ket pont koze
				if (isPointOnLine(points.at(0), points.at(1), vec2(0.0f, 0.0f)) && !equalsVec2(points.at(0), points.at(1))) {
					line1.push_back(points.at(0)); 
					line1.push_back(points.at(1));

					vonal1->countLineLength(line1);
					line1.pop_back();

					p2->setVertices(points.at(1), 0.007f);
				}
				//ez az eset az, mikor ket pont ugyanazon a koordinatakon van
				else {
					points.clear();
				}
			}
			break;
		case 3:

			//az aktualis harom pont
			printf("Points:\n");
			for (unsigned int i = 0; i < points.size(); i++) {
				printf("(%3.2f, %3.2f)\n", points.at(i).x, points.at(i).y);
			}

			circleCenter2 = CountOtherCircleCenter(points.at(1), points.at(2));

			// a case 2-ben leirt logika alapjan mukodik ez a ketto if-else is
			if (!equalsVec2(circleCenter2, vec2(-1000.0f, -1000.0f))) {
				line2 = vonal2->setVertices(circleCenter2, points.at(1), points.at(2), 1);
				p3->setVertices(points.at(2), 0.007f);
			}
			else {
				if (isPointOnLine(points.at(1), points.at(2), vec2(0.0f, 0.0f)) && !equalsVec2(points.at(1), points.at(2))) {
					line2.push_back(points.at(1));
					line2.push_back(points.at(2));

					vonal2->countLineLength(line2);
					line2.pop_back();

					p3->setVertices(points.at(2), 0.007f);
				}
				else {
					points.clear();
				}
			}

			circleCenter3 = CountOtherCircleCenter(points.at(0), points.at(2));

			if (!equalsVec2(circleCenter2, vec2(-1000.0f, -1000.0f))) {
				line3 = vonal3->setVertices(circleCenter3, points.at(2), points.at(0), 2);
			}
			else {
				if (isPointOnLine(points.at(2), points.at(0), vec2(0.0f, 0.0f)) && !equalsVec2(points.at(2), points.at(0))) {
					line3.push_back(points.at(2));
					line3.push_back(points.at(0));

					vonal3->countLineLength(line3);
					line3.pop_back();
				}
				else {
					points.clear();
				}
			}

			//a harom oldala a haromszognek
			triangle->addSide(vonal1);
			triangle->addSide(vonal2);
			triangle->addSide(vonal3);

			//a haromszog csucspontjai
			triangle->addEndPoint(points[0]);
			triangle->addEndPoint(points[1]);
			triangle->addEndPoint(points[2]);

			//result valtozo tarolja a sziriuszi haromszog(poligon) osszes vertex-et, amin majd a fulvago lefut
			std::vector<vec2> result(line1);
			result.insert(result.end(), line2.begin(), line2.end());
			result.insert(result.end(), line3.begin(), line3.end());

			//harom szog kiszamitasa
			float alpha = countAngle(points)[0];
			float beta = countAngle(points)[1];
			float gamma = countAngle(points)[2];

			//fulvagas
			triangle->setFillingTriangles(result);

			//szogek
			printf("alpha = %3.4f, beta = %3.4f, gamma = %3.4f,sum = %3.4f\n", alpha, beta, gamma, float(alpha + beta + gamma));
			//oldalhosszak
			printf("a = %3.4f, b = %3.4f, c = %3.4f\n", vonal1->getLineLength(), vonal2->getLineLength(), vonal3->getLineLength());

			//pontok torlese mindig
			points.clear();
			break;
		}
	}
	//kepernyo invalidalasa
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

#ifdef COLOR
	//a szin valtoztatasahoz szukseges
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	if(time % 25 == 0)
		glutPostRedisplay();

	if (time > 100000)
		time = 0;
#endif 
}
