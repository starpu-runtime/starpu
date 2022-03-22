package fr.labri.hpccloud.yarn.examples.pingpong;

import fr.labri.hpccloud.yarn.MasterSlaveObject;
import fr.labri.hpccloud.yarn.Utils;

import java.io.*;
import java.net.Socket;
import java.util.Random;

public class Client extends MasterSlaveObject.Slave {
    int serverPort;

    Client(String prefix) throws IOException{
        super(prefix);
        serverPort = Integer.parseInt(Utils.getMandatoryEnv("SERVER_PORT"));
    }

    void run(PrintStream out) {
        int clientId = Integer.parseInt(slaveID);
        try {
            int t = new Random().nextInt(5000);
            out.println ("sleeping for "+t);

            Thread.sleep(t);
            Socket socket = new Socket(masterIP, serverPort);

            DataOutputStream writer = new DataOutputStream(socket.getOutputStream());
            writer.writeInt(clientId);
            writer.flush();

            DataInputStream reader = new DataInputStream(socket.getInputStream());
            int c = reader.readInt();
            int ack = Server.ack(clientId);
            if (c == ack)
                out.println(String.format("client %d: receive ack from server.", clientId));
            else
                out.println(String.format("invalid ack %d != %d.", ack, c));
            writer.close();
            reader.close();
            socket.close();
        } catch (Exception e) {
            err.println(String.format("client %d: %s", clientId, e.getMessage()));
        }
    }

    public void run() {
        run(out);
        terminate();
    }

    public static void main(String[] args) throws Exception {
        Client client = new Client("client-");
        client.run();
    }
}
