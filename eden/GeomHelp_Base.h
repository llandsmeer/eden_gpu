template <typename Real>
struct GeomHelp_Base{
	inline static Real Length(Real dx, Real dy, Real dz){
		return std::sqrt(dx*dx + dy*dy + dz*dz);
	}
	inline static Real Area(Real length, Real diam_proximal, Real diam_distal){
		// actually just the external surface area of the frustum
		// what about the end butt of dendrites (and soma which is actually large!), though?
		// do what NEURON does, ignore them and let higher-level modelling software apply corrections
		if(length == 0){
			// spherical soma or something, TODO more robust detection at parse time too!
			return M_PI * diam_distal * diam_distal;
		}
		else return (M_PI / 2.0) * (diam_proximal + diam_distal) * std::sqrt( ((diam_proximal - diam_distal) * (diam_proximal - diam_distal) / 4.0) + length * length);
	}
	inline static Real Volume(Real length, Real diam_proximal, Real diam_distal){
		if(length == 0){
			// spherical soma or something, TODO more robust detection at parse time too!
			return (M_PI / 6.0) * diam_distal * diam_distal * diam_distal;
		}
		else return (M_PI / 3.0) * length * ( diam_proximal*diam_proximal + diam_distal*diam_distal + diam_proximal*diam_distal ) / 4.0;
	}
};
extern "C" {
typedef GeomHelp_Base<float> GeomHelp;
}
