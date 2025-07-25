package com.smhrd.web.Home;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/home")
public class HomeController {

    @Autowired
    private HomeService homeService;

    // 홈 화면 - 간단 메시지 반환
    @GetMapping("/")
    public Map<String, Object> home() {
        Map<String, Object> response = new HashMap<>();
        response.put("message", "백구백구 109의 home 입니다.");
        return response;
    }

    // 로그인 처리 - 성공 시 사용자 정보 JSON 반환, 실패 시 401 에러 반환
    @PostMapping("/loginCheck")
    public ResponseEntity<?> loginCheck(@RequestParam("id") String id,
                                        @RequestParam("pw") String pw,
                                        HttpSession session) {

        HomeDTO result = homeService.login(id, pw);

        if (result != null) {
            session.setAttribute("loginId", result.getUserPhone());
            session.setAttribute("userName", result.getUserName());

            Map<String, Object> res = new HashMap<>();
            res.put("userName", result.getUserName());
            res.put("userPhone", result.getUserPhone());

            if ("admin".equals(id) && "admin".equals(pw)) {
                session.setAttribute("role", "admin");
                res.put("role", "admin");
            } else {
                session.setAttribute("role", "user");
                res.put("role", "user");
            }

            return ResponseEntity.ok(res);
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "아이디 또는 비밀번호가 올바르지 않습니다."));
        }
    }

    // 사용자 페이지 접근 권한 체크용 API
    @GetMapping("/userPage")
    public ResponseEntity<?> userPage(HttpSession session) {
        String role = (String) session.getAttribute("role");

        if ("user".equals(role)) {
            return ResponseEntity.ok(Map.of("message", "사용자 페이지 접근 허용"));
        } else {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "접근 권한이 없습니다."));
        }
    }

    // 관리자 페이지 접근 권한 체크용 API
    @GetMapping("/adminPage")
    public ResponseEntity<?> adminPage(HttpSession session) {
        String role = (String) session.getAttribute("role");

        if ("admin".equals(role)) {
            return ResponseEntity.ok(Map.of("message", "관리자 페이지 접근 허용"));
        } else {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "접근 권한이 없습니다."));
        }
    }
}
