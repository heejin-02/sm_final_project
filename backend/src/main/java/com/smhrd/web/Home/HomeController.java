package com.smhrd.web.Home;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/home")
public class HomeController {

    @Autowired
    private HomeService homeService;

    @GetMapping("/")
    public String home() {
        return "home";
    }

    @PostMapping("/loginCheck")
    public String loginCheck(@RequestParam("id") String id,
            @RequestParam("pw") String pw,
            HttpSession session) {

        HomeDTO result = homeService.login(id, pw);

        if (result != null) {
            session.setAttribute("loginId", result.getUserPhone());
            session.setAttribute("userName", result.getUserName());

            if ("admin".equals(id) && "admin".equals(pw)) {
                session.setAttribute("role", "admin");
                return "redirect:/admin";
            }

            session.setAttribute("role", "user");
            return "redirect:/userPage";
        } else {
            return "redirect:/?error=true";
        }
    }

    @GetMapping("/userPage")
    public String userPage(HttpSession session) {
        String role = (String) session.getAttribute("role");
        return "user".equals(role) ? "userPage" : "redirect:/?unauthorized=true";
    }

    @GetMapping("/adminPage")
    public String adminPage(HttpSession session) {
        String role = (String) session.getAttribute("role");

        if ("admin".equals(role)) {
            return "adminPage";
        } else {
            return "redirect:/?unauthorized=true";
        }
    }

}
