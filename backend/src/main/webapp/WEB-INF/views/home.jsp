<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>홈</title>
</head>
<body>
	<h1>백구백구 109의 home 입니다.</h1>

	<!-- 에러 메시지 출력 -->
	<c:if test="${param.error == 'true'}">
		<p style="color:red;">아이디 또는 비밀번호가 올바르지 않습니다.</p>
	</c:if>
	<c:if test="${param.unauthorized == 'true'}">
		<p style="color:red;">접근 권한이 없습니다.</p>
	</c:if>

	<div>
		<button>편안한모드</button>
		<button>음성 안내 듣기</button>
		<button>알림 받기</button>
	</div>

	<div>
		<form action="loginCheck" method="post">
			<ul class="">
				<li>
					<span class="">아이디</span>
					<input type="text" name="id" placeholder="아이디를 입력해주세요" class="ipt_tt" maxlength="100" required>
				</li>
				<li>
					<span class="">비밀번호</span>
					<div class="po_rel">
						<input type="password" name="pw" placeholder="비밀번호를 입력해주세요" class="ipt_tt" maxlength="200" required>						
					</div>
					<div class="chkbox_div">
						<input type="checkbox" name="id_save" id="id_save" value="id_save" <c:if test="${not empty savedId}">checked</c:if>>
						<label for="id_save" class="chk_btn"></label>
						<input type="submit" value="로그인" class="bttn bttn_sbm2">
						<label for="id_save" class="chk_txt">아이디 기억하기</label>
						<a href="find" class="ml-auto">비밀번호를 잊으셨나요?</a>
					</div>						
				</li>
				<li>
					<a>해충 때문에 고민이신가요? 든든하고 귀여운 백구가 밭을 지켜드립니다. 카메라 실시간 감지로 해충의 움직임을 10초만에 탐지해서 9초만에 알려드려요!</a>	
				</li>
			</ul>
		</form>
	</div>
</body>
</html>
