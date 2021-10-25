import sys

n = int(sys.argv[1])
print(f"""<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"  xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2.1.xsd" id="net_C51A">
    <include href="C51A.cell.nml"/>
    <pulseGenerator id="iclamp0" delay="0ms" duration="0ms" amplitude="0nA"/>
    <network id="network_of_C51A" type="networkWithTemperature" temperature="37degC">
        <population id="population_of_C51A" component="C51A" size="{n}" type="populationList">""")
for i in range(n):
    print(f'            <instance id="{i}"> <location x="{i*100}." y="0." z="0."/> </instance>')
print("""        </population>
        <inputList id="Iclamp0" population="population_of_C51A" component="iclamp0">
            <input id="0" target="../population_of_C51A/0/C51A" destination="synapses" segmentId="0"/>
        </inputList>
    </network>
</neuroml>""")
