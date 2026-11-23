import Darwin
import Foundation

public enum ControlSocket {
    /// Sends one command line to a ps5slop control socket. Returns false when
    /// the socket is missing or refuses the connection.
    public static func send(_ command: String, to path: String) -> Bool {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(path.utf8)
        let capacity = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathBytes.count < capacity else { return false }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: pathBytes)
            raw[pathBytes.count] = 0
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let connected = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, len) }
        }
        guard connected == 0 else { return false }

        let line = Array((command + "\n").utf8)
        let written = line.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        return written == line.count
    }
}
