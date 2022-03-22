package fr.labri.hpccloud.yarn.examples.pingpong;

import fr.labri.hpccloud.yarn.MasterSlaveObject;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CreateFlag;
import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.Path;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.EnumSet;

public class Server extends MasterSlaveObject.Master implements Runnable {
    int port;
    int nbClients;
    ServerSocket socket;

    public Server(String prefix, int nbClients, int port) throws IOException {
        super(prefix);
        this.port = port;
        this.nbClients = nbClients;
        socket = new ServerSocket(port);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        socket.close();
    }

    public static int ack(int clientId) {
        return (3*clientId^19);
    }

    public void run(PrintStream out) {
        out.println("server started");
        try {
            Thread[] tasks = new Thread[nbClients];

            for (int i = 0; i < nbClients; i++) {
                try {
                    final Socket client = socket.accept();

                    tasks[i] = new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                out.println("New connection from " + client.getInetAddress());
                                DataInputStream reader = new DataInputStream(client.getInputStream());
                                int clientIndex = reader.readInt();
                                out.println("client : " + clientIndex + " from " + client.getInetAddress());

                                DataOutputStream writer = new DataOutputStream(client.getOutputStream());
                                writer.writeInt(ack(clientIndex));
                                writer.flush();

                                reader.close();
                                writer.close();
                                client.close();
                            } catch (IOException e) {
                                out.println("error:" + e.getMessage());
                            }
                        }
                    });
                    tasks[i].start();

                } catch (IOException e) {
                    out.println("error:" + e.getMessage());
                }
            }
            for (int i = 0; i < nbClients; i++) {
                try {
                    tasks[i].join();
                } catch (InterruptedException e) {
                    out.println("error:" + e.getMessage());
                }
            }
        } catch (Exception e) {
            err.println(e.getMessage());
        }
        out.println("terminate server run()");
    }

    @Override
    public void run() {
        try {
            run(out);
            terminate();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    public static Server startServer(int nbClients, int port) throws IOException {
        Server server = new Server("server", nbClients, port);
        new Thread(server).start();
        return server;
    }

    public static void main(String[] args) throws Exception {
        int nbClients = Integer.parseInt(System.getenv("NB_CLIENTS"));
        int port = Integer.parseInt(System.getenv("SERVER_PORT"));
        Server.startServer(nbClients, port);
    }
}
