package com.smhrd.web.Home;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@CrossOrigin(origins = "http://localhost:5173", allowCredentials = "true")
public class HomeController {

    @Autowired
    private HomeService homeService;

    @GetMapping("/")
    public String home() {
        return "home";
    }

    @PostMapping("/login")
    public ResponseEntity<HomeDTO> login(@RequestBody HomeDTO dto, HttpSession session) {
        HomeDTO user = homeService.login(dto.getUserPhone(), dto.getUserPw());
        if (user == null) {
            return ResponseEntity.status(401).build();
        }
        String role = "admin".equals(dto.getUserPhone()) ? "admin" : "user";
        session.setAttribute("role", role);
        session.setAttribute("userName", user.getUserName());
        user.setUserPw(null);
        user.setRole(role);
        return ResponseEntity.ok(user);
    }

    @GetMapping("/me")
    public ResponseEntity<HomeDTO> me(HttpSession session) {
        String phone = (String) session.getAttribute("loginId");
        if (phone == null)
            return ResponseEntity.status(401).build();
        HomeDTO user = new HomeDTO();
        user.setUserPhone(phone);
        user.setUserName((String) session.getAttribute("userName"));
        user.setRole((String) session.getAttribute("role"));
        return ResponseEntity.ok(user);
    }

}
