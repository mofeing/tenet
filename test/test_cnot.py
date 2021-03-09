import circuit
import gate
import ring


def main():
    # Parameters
    n = 20
    chi = 64

    # Generate circuit
    circ = circuit.Circuit(n)
    for i in range(n):
        circ.add_gate(i, gate.H())

    for i in range(n/2):
        circ.add_gate((2*i, 2*i+1), gate.CX())

    for i in range(n):
        circ.add_gate(i, gate.H())

    # Simulation
    sim = ring.Ring(n, chi)
    sim.run(circ)


if __name__ == '__main__':
    main()
